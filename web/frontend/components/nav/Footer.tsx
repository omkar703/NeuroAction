import React from 'react';
import Link from 'next/link';

export default function Footer() {
  return (
    <footer className="w-full border-t border-[#DCDDD5] bg-[#F7F6F2] py-10 mt-auto text-sm text-[#6F716B]">
      <div className="max-w-6xl mx-auto px-6 lg:px-8">
        <div className="flex flex-col md:flex-row md:items-baseline md:justify-between gap-6 pb-8 border-b border-[#DCDDD5]">
          <div>
            <div className="font-serif text-lg text-[#252722] font-semibold tracking-tight">
              NeuroAction
            </div>
            <p className="mt-1 text-xs text-[#6F716B] max-w-md leading-relaxed">
              An offline multimodal research platform investigating pre-movement scalp EEG goal decoding and 3D limb trajectory forecasting on OpenNeuro dataset ds006840.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-xs">
            <Link href="/" className="text-[#252722] hover:text-[#455D4A] transition-colors">
              Overview
            </Link>
            <Link href="/science" className="text-[#252722] hover:text-[#455D4A] transition-colors">
              Science
            </Link>
            <Link href="/model-lab" className="text-[#252722] hover:text-[#455D4A] transition-colors">
              Model Lab
            </Link>
            <Link href="/digital-twin" className="text-[#252722] hover:text-[#455D4A] transition-colors">
              3D Twin
            </Link>
            <Link href="/explainability" className="text-[#252722] hover:text-[#455D4A] transition-colors">
              Explainability
            </Link>
            <Link href="/evidence" className="text-[#252722] hover:text-[#455D4A] transition-colors">
              Evidence
            </Link>
          </div>
        </div>

        <div className="pt-6 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-[11px] text-[#6F716B]">
          <p>
            Status: Software verified; existing-data evaluation exploratory. Non-clinical research prototype.
          </p>
          <div className="flex items-center gap-4">
            <span>OpenNeuro ds006840</span>
            <span>&bull;</span>
            <span>269 Audited Reach Trials</span>
          </div>
        </div>
      </div>
    </footer>
  );
}
