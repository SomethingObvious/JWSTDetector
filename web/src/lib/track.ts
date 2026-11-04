/**
 * An $effect reruns when something it read changes. This reads values the effect
 * has to react to but doesn't otherwise use, like the theme behind the CSS colours
 * it paints with.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars -- passing the values in is the read
export function track(..._values: unknown[]): void {
  // Nothing left to do once the arguments have been read.
}
