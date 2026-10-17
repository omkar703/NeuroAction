'use client';

import React, { useMemo } from 'react';
import * as THREE from 'three';

export interface HandModelProps {
  position: [number, number, number];
  rotation?: [number, number, number];
  color?: string;
}

interface FingerProps {
  basePosition: [number, number, number];
  baseRotation?: [number, number, number];
  lengths: [number, number, number]; // proximal, middle, distal
  radius: number;
  curl?: [number, number, number]; // rotX per segment
  spreadY?: number; // rotY spread
  material: THREE.Material;
}

interface ThumbProps {
  basePosition: [number, number, number];
  baseRotation: [number, number, number];
  lengths: [number, number]; // proximal, distal
  radius: number;
  curl?: [number, number];
  material: THREE.Material;
}

/**
 * Articulated 3-segment finger (proximal, middle, distal)
 */
function Finger({
  basePosition,
  baseRotation = [0, 0, 0],
  lengths,
  radius,
  curl = [0.22, 0.32, 0.22],
  spreadY = 0,
  material,
}: FingerProps) {
  const [l1, l2, l3] = lengths;
  const [c1, c2, c3] = curl;

  return (
    <group position={basePosition} rotation={[baseRotation[0], baseRotation[1] + spreadY, baseRotation[2]]}>
      {/* Knuckle Joint */}
      <mesh material={material}>
        <sphereGeometry args={[radius * 1.15, 12, 12]} />
      </mesh>

      {/* Proximal Segment */}
      <group rotation={[c1, 0, 0]}>
        <mesh position={[0, l1 / 2, 0]} material={material}>
          <cylinderGeometry args={[radius * 0.95, radius, l1, 12]} />
        </mesh>

        {/* Middle Joint */}
        <group position={[0, l1, 0]}>
          <mesh material={material}>
            <sphereGeometry args={[radius * 1.05, 12, 12]} />
          </mesh>

          {/* Middle Segment */}
          <group rotation={[c2, 0, 0]}>
            <mesh position={[0, l2 / 2, 0]} material={material}>
              <cylinderGeometry args={[radius * 0.85, radius * 0.95, l2, 12]} />
            </mesh>

            {/* Distal Joint */}
            <group position={[0, l2, 0]}>
              <mesh material={material}>
                <sphereGeometry args={[radius * 0.92, 12, 12]} />
              </mesh>

              {/* Distal Segment */}
              <group rotation={[c3, 0, 0]}>
                <mesh position={[0, l3 / 2, 0]} material={material}>
                  <cylinderGeometry args={[radius * 0.72, radius * 0.85, l3, 12]} />
                </mesh>
                {/* Fingertip cap */}
                <mesh position={[0, l3, 0]} material={material}>
                  <sphereGeometry args={[radius * 0.72, 12, 12]} />
                </mesh>
              </group>
            </group>
          </group>
        </group>
      </group>
    </group>
  );
}

/**
 * Articulated 2-segment thumb angled outward
 */
function Thumb({
  basePosition,
  baseRotation,
  lengths,
  radius,
  curl = [0.28, 0.35],
  material,
}: ThumbProps) {
  const [l1, l2] = lengths;
  const [c1, c2] = curl;

  return (
    <group position={basePosition} rotation={baseRotation}>
      {/* Base joint */}
      <mesh material={material}>
        <sphereGeometry args={[radius * 1.2, 12, 12]} />
      </mesh>

      {/* Proximal segment */}
      <group rotation={[c1, 0, 0]}>
        <mesh position={[0, l1 / 2, 0]} material={material}>
          <cylinderGeometry args={[radius * 0.9, radius * 1.05, l1, 12]} />
        </mesh>

        {/* Interphalangeal joint */}
        <group position={[0, l1, 0]}>
          <mesh material={material}>
            <sphereGeometry args={[radius * 1.0, 12, 12]} />
          </mesh>

          {/* Distal segment */}
          <group rotation={[c2, 0, 0]}>
            <mesh position={[0, l2 / 2, 0]} material={material}>
              <cylinderGeometry args={[radius * 0.75, radius * 0.9, l2, 12]} />
            </mesh>
            <mesh position={[0, l2, 0]} material={material}>
              <sphereGeometry args={[radius * 0.75, 12, 12]} />
            </mesh>
          </group>
        </group>
      </group>
    </group>
  );
}

/**
 * Procedural Articulated Hand Model
 * Uses standard primitives and standard materials with skin tone
 * Proportions (Three.js units, scaled from Leap Motion mm):
 * - Palm: 0.08 × 0.05 × 0.03
 * - Wrist: flattened cylinder below palm
 * - Forearm: tapered cylinder extending downwards
 * - Fingers: Index, Middle, Ring (0.04, 0.03, 0.02), Pinky (0.03, 0.02, 0.015), Thumb (0.035, 0.025)
 */
export default function HandModel({
  position,
  rotation = [0, 0, 0],
  color = '#e2b897',
}: HandModelProps) {
  // Shared material across hand meshes
  const handMaterial = useMemo(
    () =>
      new THREE.MeshStandardMaterial({
        color: new THREE.Color(color),
        roughness: 0.7,
        metalness: 0.0,
      }),
    [color]
  );

  return (
    <group position={position} rotation={rotation}>
      {/* PALM: Rounded box 0.08 × 0.05 × 0.03 */}
      <mesh position={[0, 0, 0]} material={handMaterial}>
        <boxGeometry args={[0.08, 0.05, 0.03]} />
      </mesh>

      {/* WRIST: Flattened cylinder below palm */}
      <mesh position={[0, -0.035, 0]} scale={[1, 1, 0.72]} material={handMaterial}>
        <cylinderGeometry args={[0.032, 0.03, 0.02, 16]} />
      </mesh>

      {/* FOREARM: Tapered cylinder extending downward */}
      <mesh position={[0, -0.115, -0.005]} scale={[1, 1, 0.85]} material={handMaterial}>
        <cylinderGeometry args={[0.028, 0.022, 0.14, 16]} />
      </mesh>

      {/* 5 ARTICULATED FINGERS */}

      {/* 1. Pinky Finger: 0.03, 0.02, 0.015 */}
      <Finger
        basePosition={[-0.028, 0.025, 0.002]}
        lengths={[0.03, 0.02, 0.015]}
        radius={0.006}
        spreadY={-0.08}
        curl={[0.26, 0.36, 0.28]}
        material={handMaterial}
      />

      {/* 2. Ring Finger: 0.04, 0.03, 0.02 */}
      <Finger
        basePosition={[-0.009, 0.025, 0.003]}
        lengths={[0.04, 0.03, 0.02]}
        radius={0.0068}
        spreadY={-0.02}
        curl={[0.23, 0.33, 0.24]}
        material={handMaterial}
      />

      {/* 3. Middle Finger: 0.042, 0.03, 0.02 */}
      <Finger
        basePosition={[0.01, 0.025, 0.003]}
        lengths={[0.042, 0.03, 0.02]}
        radius={0.007}
        spreadY={0.01}
        curl={[0.22, 0.31, 0.22]}
        material={handMaterial}
      />

      {/* 4. Index Finger: 0.04, 0.03, 0.02 */}
      <Finger
        basePosition={[0.029, 0.025, 0.002]}
        lengths={[0.04, 0.03, 0.02]}
        radius={0.0068}
        spreadY={0.06}
        curl={[0.2, 0.29, 0.2]}
        material={handMaterial}
      />

      {/* 5. Thumb: 2 segments, 0.035, 0.025, angled outward */}
      <Thumb
        basePosition={[0.042, -0.006, 0.006]}
        baseRotation={[0.35, 0.45, -0.72]}
        lengths={[0.035, 0.025]}
        radius={0.0076}
        curl={[0.25, 0.32]}
        material={handMaterial}
      />
    </group>
  );
}
