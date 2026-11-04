/** Marker outline for a catalogue's sources, centred on 0, 0 and about 10 px across. */
export function catShape(cat: string): string {
  if (cat === 'SIMBAD') {
    return 'M-5 0 A5 5 0 1 0 5 0 A5 5 0 1 0 -5 0';
  }
  if (cat === 'NED') {
    return 'M-4.5 -4.5 H4.5 V4.5 H-4.5 Z';
  }
  if (cat === 'Gaia DR3') {
    return 'M-5 0 H5 M0 -5 V5';
  }
  return 'M0 -5.5 L5.5 0 L0 5.5 L-5.5 0 Z';
}

/** The catalogue's own page for a source, where it has one. */
export function sourceLink(cat: string, name: string): string | null {
  if (cat === 'SIMBAD') {
    return `https://simbad.cds.unistra.fr/simbad/sim-id?Ident=${encodeURIComponent(name)}`;
  }
  if (cat === 'NED') {
    return `https://ned.ipac.caltech.edu/byname?objname=${encodeURIComponent(name)}`;
  }
  if (cat === 'Gaia DR3') {
    const id = name.replace(/^Gaia DR3\s*/iu, '');
    return /^\d+$/u.test(id)
      ? `https://vizier.cds.unistra.fr/viz-bin/VizieR-5?-source=I/355/gaiadr3&Source=${id}`
      : null;
  }
  return null;
}

/** "SIMBAD, NED or Gaia DR3". */
export function orList(names: string[]): string {
  if (names.length < 2) {
    return names[0] ?? '';
  }
  return `${names.slice(0, -1).join(', ')} or ${names.at(-1) ?? ''}`;
}
