export const colors = {
  // Semantic signal channels (academic, high-contrast palette)
  eegCyan: '#2563EB',         // EEG signal / neural traces (clear scholarly blue)
  goalBlue: '#1D4ED8',        // decoded / instructed goal
  trajGreen: '#2E6B47',       // recorded physical trajectory (academic forest green)
  forecastViolet: '#6B46C1',  // predicted trajectory (subdued violet)
  errorOrange: '#B45309',     // evaluation error / warning (warm ochre)
  grayNeutral: '#6F716B',     // baseline / unvalidated (neutral slate)
  navyBg: '#F7F6F2',          // primary background (warm ivory)

  // Quiet Academic Editorial Design Tokens
  ivory: '#F7F6F2',           // primary page background
  ivorySubtle: '#FAF9F6',     // subtle raised ivory
  surface: '#FFFFFF',         // clean white card/table surface
  surfaceInset: '#F0EFE9',     // slightly darker warm neutral for recessed areas
  charcoal: '#252722',        // primary reading text
  muted: '#6F716B',           // secondary/caption text
  forest: '#455D4A',          // primary accent (academic forest green)
  forestHover: '#364A3B',     // darker forest green on hover
  forestSubtle: '#EBF0EC',    // very soft green tint for badges
  border: '#DCDDD5',          // hairline divider / table border
  borderStrong: '#BCBEB5',    // stronger divider
} as const;

export type ColorKey = keyof typeof colors;
