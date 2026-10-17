'use client';

import React, { useRef, useEffect, useMemo } from 'react';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import HandModel from './HandModel';
import { TrialData } from '@/lib/types';

export interface Workspace3DProps {
  trialData: TrialData | null;
  currentTime: number;
  showGrid?: boolean;
  showAxes?: boolean;
  showForecast?: boolean;
  resetCameraSignal?: number;
}

// Physical positions converted to Three.js units (1 mm = 0.01 units)
const TARGET_LEFT_UNITS: [number, number, number] = [-1.88, 1.5, 0.53];
const TARGET_RIGHT_UNITS: [number, number, number] = [1.88, 1.49, 0.51];
const ORIGIN_UNITS: [number, number, number] = [0.0, 1.5, 0.5];

/**
 * Interpolate 3D position in mm at a given timestamp
 */
function interpolatePositionMm(
  trialData: TrialData | null,
  currentTimeMs: number
): { x: number; y: number; z: number; vx: number } {
  if (
    !trialData ||
    !trialData.times_ms ||
    trialData.times_ms.length === 0 ||
    !trialData.hand_x ||
    trialData.hand_x.length === 0
  ) {
    return { x: 0, y: 150, z: 50, vx: 0 };
  }

  const { times_ms, hand_x, hand_y, hand_z } = trialData;
  const len = Math.min(times_ms.length, hand_x.length, hand_y.length, hand_z.length);

  if (currentTimeMs <= times_ms[0]) {
    return { x: hand_x[0], y: hand_y[0], z: hand_z[0], vx: 0 };
  }
  if (currentTimeMs >= times_ms[len - 1]) {
    return { x: hand_x[len - 1], y: hand_y[len - 1], z: hand_z[len - 1], vx: 0 };
  }

  // Find bounding sample indices
  let idx = 0;
  for (let i = 0; i < len - 1; i++) {
    if (currentTimeMs >= times_ms[i] && currentTimeMs <= times_ms[i + 1]) {
      idx = i;
      break;
    }
  }

  const t0 = times_ms[idx];
  const t1 = times_ms[idx + 1];
  const dt = Math.max(t1 - t0, 0.001);
  const alpha = Math.max(0, Math.min(1, (currentTimeMs - t0) / dt));

  const x = hand_x[idx] + alpha * (hand_x[idx + 1] - hand_x[idx]);
  const y = hand_y[idx] + alpha * (hand_y[idx + 1] - hand_y[idx]);
  const z = hand_z[idx] + alpha * (hand_z[idx + 1] - hand_z[idx]);
  const vx = (hand_x[idx + 1] - hand_x[idx]) / dt;

  return { x, y, z, vx };
}

/**
 * Main 3D Workspace Scene
 */
export default function Workspace3D({
  trialData,
  currentTime,
  showGrid = true,
  showAxes = true,
  showForecast = true,
  resetCameraSignal = 0,
}: Workspace3DProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);

  // Reset camera view on signal
  useEffect(() => {
    if (resetCameraSignal > 0 && controlsRef.current) {
      if (controlsRef.current.target && controlsRef.current.object) {
        controlsRef.current.target.set(0, 1.5, 0);
        controlsRef.current.object.position.set(0, 3, 4);
        controlsRef.current.update();
      }
    }
  }, [resetCameraSignal]);

  // Compute interpolated hand position in physical mm & Three.js units
  const currentPosMm = useMemo(
    () => interpolatePositionMm(trialData, currentTime),
    [trialData, currentTime]
  );

  const currentHandUnits: [number, number, number] = useMemo(
    () => [currentPosMm.x * 0.01, currentPosMm.y * 0.01, currentPosMm.z * 0.01],
    [currentPosMm]
  );

  // Natural banking rotation along lateral trajectory
  const handRotation: [number, number, number] = useMemo(() => {
    const roll = Math.max(-0.25, Math.min(0.25, -currentPosMm.vx * 0.15));
    return [0, 0, roll];
  }, [currentPosMm.vx]);

  // Observed trajectory points in Three.js units
  const observedPoints = useMemo<[number, number, number][]>(() => {
    if (!trialData || !trialData.hand_x || trialData.hand_x.length < 2) return [];
    const len = Math.min(
      trialData.hand_x.length,
      trialData.hand_y.length,
      trialData.hand_z.length
    );
    const pts: [number, number, number][] = [];
    for (let i = 0; i < len; i++) {
      pts.push([
        trialData.hand_x[i] * 0.01,
        trialData.hand_y[i] * 0.01,
        trialData.hand_z[i] * 0.01,
      ]);
    }
    return pts;
  }, [trialData]);

  // Forecast trajectory points in Three.js units
  const forecastPoints = useMemo<[number, number, number][]>(() => {
    if (!trialData) return [];

    // Parse horizon_path if present
    if (trialData.horizon_path && trialData.horizon_path.length >= 2) {
      return trialData.horizon_path.map((pt) => {
        if (Array.isArray(pt)) {
          return [(pt[0] ?? 0) * 0.01, (pt[1] ?? 0) * 0.01, (pt[2] ?? 0) * 0.01];
        }
        if (pt && typeof pt === 'object') {
          return [(pt.x ?? 0) * 0.01, (pt.y ?? 0) * 0.01, (pt.z ?? 0) * 0.01];
        }
        return [0, 0, 0];
      });
    }

    // Fallback: connect hand start to predicted endpoint if present
    if (trialData.predicted_endpoint && Array.isArray(trialData.predicted_endpoint)) {
      const ep = trialData.predicted_endpoint;
      const startX = trialData.hand_x?.[0] ?? 0;
      const startY = trialData.hand_y?.[0] ?? 150;
      const startZ = trialData.hand_z?.[0] ?? 50;
      return [
        [startX * 0.01, startY * 0.01, startZ * 0.01],
        [(ep[0] ?? 0) * 0.01, (ep[1] ?? 150) * 0.01, (ep[2] ?? 50) * 0.01],
      ];
    }

    return [];
  }, [trialData]);

  // Forecasted endpoint coordinates in units
  const forecastEndpointUnits: [number, number, number] | null = useMemo(() => {
    if (!trialData?.predicted_endpoint || !Array.isArray(trialData.predicted_endpoint)) {
      return null;
    }
    const ep = trialData.predicted_endpoint;
    return [(ep[0] ?? 0) * 0.01, (ep[1] ?? 150) * 0.01, (ep[2] ?? 50) * 0.01];
  }, [trialData]);

  return (
    <>
      {/* SCENE BACKGROUND & FOG */}
      <color attach="background" args={['#F5F8F5']} />
      <fog attach="fog" args={['#F5F8F5', 6, 20]} />

      {/* CAMERA CONTROLS: default pos (0, 3, 4), target (0, 1.5, 0) */}
      <OrbitControls
        ref={controlsRef}
        makeDefault
        target={[0, 1.5, 0]}
        minDistance={1.2}
        maxDistance={12}
        enableDamping
        dampingFactor={0.08}
      />

      {/* AMBIENT & DIRECTIONAL LIGHTING */}
      <ambientLight intensity={1.1} />
      <directionalLight position={[5, 10, 6]} intensity={1.3} castShadow />
      <directionalLight position={[-4, 6, -4]} intensity={0.5} color="#e6f4ea" />
      <pointLight position={[0, 3, 1]} intensity={0.6} color="#ffffff" />

      {/* FLOOR GRID (Light sage & white theme) */}
      {showGrid && (
        <group position={[0, 0, 0]}>
          <gridHelper args={[8, 32, '#A8C2B0', '#DCE8DF']} />
          {/* Subtle desk surface boundary */}
          <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]}>
            <planeGeometry args={[7.8, 7.8]} />
            <meshBasicMaterial color="#EBF2EC" transparent opacity={0.85} />
          </mesh>
        </group>
      )}

      {/* SPATIAL AXES HELPER & LABELS */}
      {showAxes && (
        <group position={[0, 0.001, 0]}>
          <axesHelper args={[1.6]} />
          <Html position={[1.7, 0.05, 0]} center distanceFactor={4.5} pointerEvents="none">
            <span className="text-[8px] font-body font-semibold px-1.5 py-0.5 rounded-full bg-[#FFFFFF]/90 text-red-600 border border-red-200 shadow-xs select-none">
              +X Lateral
            </span>
          </Html>
          <Html position={[0, 1.7, 0]} center distanceFactor={4.5} pointerEvents="none">
            <span className="text-[8px] font-body font-semibold px-1.5 py-0.5 rounded-full bg-[#FFFFFF]/90 text-[#3A5244] border border-[#3A5244]/30 shadow-xs select-none">
              +Y Height
            </span>
          </Html>
          <Html position={[0, 0.05, 1.7]} center distanceFactor={4.5} pointerEvents="none">
            <span className="text-[8px] font-body font-semibold px-1.5 py-0.5 rounded-full bg-[#FFFFFF]/90 text-blue-600 border border-blue-200 shadow-xs select-none">
              +Z Depth
            </span>
          </Html>
        </group>
      )}

      {/* ORIGIN / RESTING POSITION MARKER */}
      <group position={ORIGIN_UNITS}>
        <mesh>
          <sphereGeometry args={[0.03, 16, 16]} />
          <meshStandardMaterial color="#94a3b8" roughness={0.4} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.045, 0.055, 24]} />
          <meshBasicMaterial color="#A8C2B0" side={THREE.DoubleSide} transparent opacity={0.6} />
        </mesh>
      </group>

      {/* LEFT TARGET: Blue (-1.88, 1.5, 0.53) */}
      <group position={TARGET_LEFT_UNITS}>
        <mesh>
          <sphereGeometry args={[0.075, 24, 24]} />
          <meshStandardMaterial
            color="#3b82f6"
            emissive="#1d4ed8"
            emissiveIntensity={0.35}
            roughness={0.3}
            metalness={0.1}
          />
        </mesh>
        {/* Holographic Target Outer Ring */}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.095, 0.115, 32]} />
          <meshBasicMaterial color="#60a5fa" side={THREE.DoubleSide} transparent opacity={0.7} />
        </mesh>
        {/* Label */}
        <Html position={[0, 0.12, 0]} center distanceFactor={4.5} pointerEvents="none">
          <div className="px-2 py-0.5 rounded-full text-[9px] font-semibold bg-[#FFFFFF]/90 text-[#2563EB] border border-[#2563EB]/25 shadow-xs whitespace-nowrap font-body select-none">
            Left Target
          </div>
        </Html>
      </group>

      {/* RIGHT TARGET: Amber (1.88, 1.49, 0.51) */}
      <group position={TARGET_RIGHT_UNITS}>
        <mesh>
          <sphereGeometry args={[0.075, 24, 24]} />
          <meshStandardMaterial
            color="#f59e0b"
            emissive="#b45309"
            emissiveIntensity={0.35}
            roughness={0.3}
            metalness={0.1}
          />
        </mesh>
        {/* Holographic Target Outer Ring */}
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.095, 0.115, 32]} />
          <meshBasicMaterial color="#fbbf24" side={THREE.DoubleSide} transparent opacity={0.7} />
        </mesh>
        {/* Label */}
        <Html position={[0, 0.12, 0]} center distanceFactor={4.5} pointerEvents="none">
          <div className="px-2 py-0.5 rounded-full text-[9px] font-semibold bg-[#FFFFFF]/90 text-[#B45309] border border-[#B45309]/25 shadow-xs whitespace-nowrap font-body select-none">
            Right Target
          </div>
        </Html>
      </group>

      {/* OBSERVED TRAJECTORY PATH: Deep forest green Line (#2E6B47) */}
      {observedPoints.length >= 2 && (
        <Line
          points={observedPoints}
          color="#2E6B47"
          lineWidth={3.5}
          opacity={0.92}
          transparent
        />
      )}

      {/* FORECAST TRAJECTORY PATH: Violet Line (#8b5cf6) */}
      {showForecast && forecastPoints.length >= 2 && (
        <>
          <Line
            points={forecastPoints}
            color="#8b5cf6"
            lineWidth={2.8}
            dashed
            dashScale={12}
            dashSize={0.4}
            gapSize={0.25}
          />
          {forecastEndpointUnits && (
            <mesh position={forecastEndpointUnits}>
              <sphereGeometry args={[0.045, 16, 16]} />
              <meshStandardMaterial
                color="#8b5cf6"
                emissive="#7c3aed"
                emissiveIntensity={0.6}
                transparent
                opacity={0.85}
              />
            </mesh>
          )}
        </>
      )}

      {/* ARTICULATED PROCEDURAL HAND MODEL */}
      <HandModel position={currentHandUnits} rotation={handRotation} />

      {/* DYNAMIC COORDINATE READOUT (Compact Micro Badge floating above the Hand) */}
      <Html
        position={[currentHandUnits[0], currentHandUnits[1] + 0.12, currentHandUnits[2]]}
        center
        distanceFactor={4.2}
        pointerEvents="none"
      >
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-[#FFFFFF]/90 border border-[#D9D8D0] shadow-xs backdrop-blur-xs select-none whitespace-nowrap text-[9px] font-body text-[#1C1C1A]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244] animate-pulse" />
          <span className="font-semibold text-[#3A5244]">Hand</span>
          <span className="text-[#6B6B66]">{currentPosMm.x.toFixed(0)}, {currentPosMm.y.toFixed(0)}, {currentPosMm.z.toFixed(0)} mm</span>
        </div>
      </Html>
    </>
  );
}
