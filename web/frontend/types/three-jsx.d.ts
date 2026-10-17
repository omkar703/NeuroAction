/* eslint-disable @typescript-eslint/no-empty-object-type */
import type { ThreeElements } from '@react-three/fiber';

declare global {
  namespace React {
    namespace JSX {
      interface IntrinsicElements extends ThreeElements {}
    }
  }
  namespace JSX {
    interface IntrinsicElements extends ThreeElements {}
  }
}
