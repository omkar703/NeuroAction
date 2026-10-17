'use client';

import React, { useRef, useMemo } from 'react';
import { OrbitControls, Html, Line } from '@react-three/drei';
import * as THREE from 'three';
import HandModel from './HandModel';
import { TrialData } from '@/lib/types';

export interface HeroWorkspace3DProps {
  trialData: TrialData;
  currentTime: number;
}

const TARGET_LEFT_UNITS: [number, number, number] = [-1.88, 1.5, 0.53];
const TARGET_RIGHT_UNITS: [number, number, number] = [1.88, 1.49, 0.51];
const ORIGIN_UNITS: [number, number, number] = [0.0, 1.5, 0.5];

function interpolatePositionMm(trialData: TrialData, currentTimeMs: number) {
  const { times_ms, hand_x, hand_y, hand_z } = trialData;
  const len = times_ms.length;

  if (currentTimeMs <= times_ms[0]) {
    return { x: hand_x[0], y: hand_y[0], z: hand_z[0] };
  }
  if (currentTimeMs >= times_ms[len - 1]) {
    return { x: hand_x[len - 1], y: hand_y[len - 1], z: hand_z[len - 1] };
  }

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

  return {
    x: hand_x[idx] + alpha * (hand_x[idx + 1] - hand_x[idx]),
    y: hand_y[idx] + alpha * (hand_y[idx + 1] - hand_y[idx]),
    z: hand_z[idx] + alpha * (hand_z[idx + 1] - hand_z[idx]),
  };
}

export default function HeroWorkspace3D({ trialData, currentTime }: HeroWorkspace3DProps) {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const controlsRef = useRef<any>(null);

  const currentPosMm = useMemo(
    () => interpolatePositionMm(trialData, currentTime),
    [trialData, currentTime]
  );

  const currentHandUnits: [number, number, number] = useMemo(
    () => [currentPosMm.x * 0.01, currentPosMm.y * 0.01, currentPosMm.z * 0.01],
    [currentPosMm]
  );

  // Observed trajectory points up to current time
  const observedPoints = useMemo(() => {
    const pts: [number, number, number][] = [];
    const len = trialData.times_ms.length;
    for (let i = 0; i < len; i++) {
      if (trialData.times_ms[i] <= currentTime) {
        pts.push([
          trialData.hand_x[i] * 0.01,
          trialData.hand_y[i] * 0.01,
          trialData.hand_z[i] * 0.01,
        ]);
      } else {
        break;
      }
    }
    pts.push(currentHandUnits);
    return pts;
  }, [trialData, currentTime, currentHandUnits]);

  // Forecast continuation points
  const forecastPoints = useMemo<[number, number, number][]>(() => {
    const continuation = (trialData as { forecast_continuation?: { x: number; y: number; z: number }[] })
      ?.forecast_continuation;
    if (!Array.isArray(continuation)) return [];
    return continuation.map(
      (p) => [p.x * 0.01, p.y * 0.01, p.z * 0.01] as [number, number, number]
    );
  }, [trialData]);

  const handRotation: [number, number, number] = useMemo(() => {
    const progress = Math.min(currentTime / 1200, 1);
    return [-0.2 + progress * 0.1, 0.35 * progress, -0.05];
  }, [currentTime]);

  return (
    <>
      <ambientLight intensity={1.5} />
      <directionalLight position={[4, 7, 5]} intensity={1.8} castShadow />
      <directionalLight position={[-4, 3, -2]} intensity={0.8} />

      <OrbitControls
        ref={controlsRef}
        target={[0.0, 1.5, 0.4]}
        enableZoom={false}
        enablePan={false}
        maxPolarAngle={Math.PI / 2.1}
        minPolarAngle={Math.PI / 4.5}
        maxAzimuthAngle={Math.PI / 3}
        minAzimuthAngle={-Math.PI / 3}
      />

      {/* Surface Desk Grid */}
      <group position={[0, 0, 0]}>
        <gridHelper args={[24, 60, '#A8C2B0', '#E2ECE4']} />
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.005, 0]}>
          <planeGeometry args={[24, 24]} />
          <meshBasicMaterial color="#F2F6F3" transparent opacity={0.85} />
        </mesh>
      </group>

      {/* Resting Origin Marker */}
      <group position={ORIGIN_UNITS}>
        <mesh>
          <sphereGeometry args={[0.025, 16, 16]} />
          <meshStandardMaterial color="#94A3B8" roughness={0.4} />
        </mesh>
      </group>

      {/* RIGHT TARGET: Primary Active Reach Target (Right Side) */}
      <group position={TARGET_RIGHT_UNITS}>
        <mesh>
          <sphereGeometry args={[0.075, 24, 24]} />
          <meshStandardMaterial
            color="#F59E0B"
            emissive="#B45309"
            emissiveIntensity={0.4}
            roughness={0.3}
          />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[0.095, 0.115, 32]} />
          <meshBasicMaterial color="#FBBF24" side={THREE.DoubleSide} transparent opacity={0.8} />
        </mesh>
        <Html position={[0, 0.11, 0]} center distanceFactor={4.5} pointerEvents="none">
          <div className="px-2 py-0.5 rounded-full text-[9px] font-semibold bg-[#FFFFFF]/90 text-[#B45309] border border-[#B45309]/25 shadow-xs whitespace-nowrap font-body select-none">
            Right Target
          </div>
        </Html>
      </group>

      {/* LEFT TARGET: Inactive / Reference Target (Softly Blurred & Dimmed) */}
      <group position={TARGET_LEFT_UNITS}>
        <mesh>
          <sphereGeometry args={[0.045, 16, 16]} />
          <meshStandardMaterial
            color="#60A5FA"
            emissive="#2563EB"
            emissiveIntensity={0.1}
            roughness={0.7}
            transparent
            opacity={0.4}
          />
        </mesh>
        <Html position={[0, 0.09, 0]} center distanceFactor={4.5} pointerEvents="none">
          <div
            style={{ filter: 'blur(1.8px)', opacity: 0.55 }}
            className="px-1.5 py-0.5 rounded-full text-[8px] font-semibold bg-[#FFFFFF]/70 text-[#2563EB] border border-[#2563EB]/20 shadow-xs whitespace-nowrap font-body select-none"
          >
            Left Target
          </div>
        </Html>
      </group>

      {/* Observed Reach Trajectory Path (Forest Green) */}
      {observedPoints.length >= 2 && (
        <Line
          points={observedPoints}
          color="#2E6B47"
          lineWidth={3.8}
          opacity={0.95}
          transparent
        />
      )}

      {/* Forecast Ridge Extrapolation Line (Violet Dashed) */}
      {forecastPoints.length >= 2 && (
        <Line
          points={forecastPoints}
          color="#8B5CF6"
          lineWidth={2.8}
          dashed
          dashScale={12}
          dashSize={0.4}
          gapSize={0.25}
        />
      )}

      {/* Articulated Procedural Hand Model */}
      <HandModel position={currentHandUnits} rotation={handRotation} />

      {/* Compact Coordinates Pill Floating Above Hand */}
      <Html
        position={[currentHandUnits[0], currentHandUnits[1] + 0.12, currentHandUnits[2]]}
        center
        distanceFactor={4.2}
        pointerEvents="none"
      >
        <div className="flex items-center gap-1.5 px-2 py-0.5 rounded-full bg-[#FFFFFF]/95 border border-[#D9D8D0] shadow-xs backdrop-blur-xs select-none whitespace-nowrap text-[9px] font-body text-[#1C1C1A]">
          <span className="w-1.5 h-1.5 rounded-full bg-[#3A5244] animate-pulse" />
          <span className="font-semibold text-[#3A5244]">Hand</span>
          <span className="text-[#6B6B66]">
            {currentPosMm.x.toFixed(0)}, {currentPosMm.y.toFixed(0)}, {currentPosMm.z.toFixed(0)} mm
          </span>
        </div>
      </Html>
    </>
  );
}
