import React from 'react';

interface NeuroActionLogoProps extends React.SVGProps<SVGSVGElement> {
  size?: number;
  className?: string;
  withBackground?: boolean;
}

export default function NeuroActionLogo({
  size = 32,
  className = '',
  withBackground = true,
  ...props
}: NeuroActionLogoProps) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 48 48"
      width={size}
      height={size}
      fill="none"
      className={className}
      aria-hidden="true"
      {...props}
    >
      {/* Squircle Background */}
      {withBackground && (
        <rect width="48" height="48" rx="12" fill="#3A5244" />
      )}

      {/* Left: Neural Cerebral Cortex */}
      <path
        d="M22 10C18 10 14.5 12 12.5 15.5C10 17 8.5 20 8.5 24C8.5 27.5 10 30.5 12.5 32C12 34.5 13.5 37 16.5 38.5C19 39.5 21.5 39 23 37"
        stroke="#F5F3EE"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Internal Sulci & Neural Gyri */}
      <path
        d="M16 15C18.5 18 17.5 22.5 13.5 24.5"
        stroke="#A8C2B0"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M12.5 21.5C16 22.5 19 20.5 19.5 18"
        stroke="#A8C2B0"
        strokeWidth="1.8"
        strokeLinecap="round"
      />
      <path
        d="M13 29C16.5 30.5 19.5 28.5 20 26"
        stroke="#A8C2B0"
        strokeWidth="1.8"
        strokeLinecap="round"
      />

      {/* Synaptic / Electrode Nodes */}
      <circle cx="13.5" cy="17" r="1.5" fill="#A8C2B0" />
      <circle cx="19.5" cy="19.5" r="1.5" fill="#A8C2B0" />
      <circle cx="14" cy="32.5" r="1.5" fill="#A8C2B0" />
      <circle cx="19" cy="30" r="1.5" fill="#A8C2B0" />

      {/* Efferent Motor Impulse Bridge (Amber Gold) */}
      <path
        d="M21 15L25 18.5L23.5 23L28 26.5"
        stroke="#FBBF24"
        strokeWidth="2.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Right: Kinematic Reaching Arm & Hand */}
      {/* Elbow */}
      <circle cx="27" cy="33" r="2" fill="#F5F3EE" />
      {/* Forearm */}
      <path
        d="M27 33L34.5 24"
        stroke="#F5F3EE"
        strokeWidth="2.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />

      {/* Wrist Joint */}
      <circle
        cx="34.5"
        cy="24"
        r="2"
        fill={withBackground ? "#3A5244" : "#FFFFFF"}
        stroke="#F5F3EE"
        strokeWidth="1.8"
      />

      {/* Reaching Fingers Extrapolating into 3D Space */}
      <path
        d="M35.5 22.5L42 17.5"
        stroke="#F5F3EE"
        strokeWidth="2.2"
        strokeLinecap="round"
      />
      <path
        d="M36 25L42 21.5"
        stroke="#F5F3EE"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <path
        d="M34 23L36.5 19.5"
        stroke="#F5F3EE"
        strokeWidth="1.8"
        strokeLinecap="round"
      />

      {/* Action Target Coordinate (Blue Target with Crosshairs) */}
      <circle cx="43" cy="13.5" r="2.2" fill="#60A5FA" />
      <path
        d="M43 8.5V11M43 16V18.5M38 13.5H40.5M45.5 13.5H48"
        stroke="#60A5FA"
        strokeWidth="1.3"
        strokeLinecap="round"
      />
    </svg>
  );
}
