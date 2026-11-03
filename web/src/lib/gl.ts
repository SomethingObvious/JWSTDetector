import { lutTexels, cmapRow } from './colormaps';
import type { Display, MapSource } from './types';
import type { Limits, TileData } from './tiles';

const VERT = `#version 300 es
void main() {
  // One triangle that covers the whole canvas.
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

const FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D u_img;
uniform highp sampler2D u_lut;
uniform highp sampler2D u_eq;
uniform highp sampler2D u_map;
uniform vec2 u_size;
uniform vec2 u_canvas;
uniform vec2 u_center;
uniform float u_zoom;
uniform float u_dpr;
uniform float u_scale;
uniform int u_channels;
uniform vec3 u_lo;
uniform vec3 u_hi;
uniform int u_mono;
uniform int u_stretch;
uniform int u_cmap;
uniform int u_invert;
uniform vec3 u_nan;
uniform int u_overlay;
uniform float u_opacity;
uniform int u_ocmap;
uniform vec2 u_orange;
uniform float u_thresh;
uniform ivec2 u_grid;
out vec4 outColor;

vec3 lut(float t, int row) {
  return texelFetch(u_lut, ivec2(int(clamp(t, 0.0, 1.0) * 255.0 + 0.5), row), 0).rgb;
}

float stretchT(float v, float lo, float hi, int plane) {
  float t = clamp((v - lo) / max(hi - lo, 1e-30), 0.0, 1.0);
  if (u_stretch == 1) return sqrt(t);
  if (u_stretch == 2) return log(1000.0 * t + 1.0) / log(1001.0);
  if (u_stretch == 3) return asinh(10.0 * t) / asinh(10.0);
  if (u_stretch == 4) return texelFetch(u_eq, ivec2(int(t * 255.0 + 0.5), plane), 0).r;
  return t;
}

float mapAt(ivec2 i) {
  return texelFetch(u_map, clamp(i, ivec2(0), u_grid - 1), 0).r;
}

void main() {
  vec2 fc = vec2(gl_FragCoord.x, u_canvas.y - gl_FragCoord.y);
  vec2 p = (fc - 0.5 * u_canvas) / u_zoom + u_center;
  if (p.x < 0.0 || p.y < 0.0 || p.x >= u_size.x || p.y >= u_size.y) {
    outColor = vec4(0.0);
    return;
  }
  vec4 raw = texelFetch(u_img, ivec2(p), 0) * u_scale;
  vec3 c;
  if (u_channels == 1 ? isnan(raw.r) : (isnan(raw.r) || isnan(raw.g) || isnan(raw.b))) {
    c = u_nan;
  } else if (u_channels == 1) {
    float t = stretchT(raw.r, u_lo.x, u_hi.x, 0);
    c = lut(u_invert == 1 ? 1.0 - t : t, u_cmap);
  } else {
    c = vec3(stretchT(raw.r, u_lo.x, u_hi.x, 0), stretchT(raw.g, u_lo.y, u_hi.y, 1), stretchT(raw.b, u_lo.z, u_hi.z, 2));
    if (u_mono == 1) {
      float t = (c.r + c.g + c.b) / 3.0;
      c = lut(u_invert == 1 ? 1.0 - t : t, u_cmap);
    } else if (u_invert == 1) {
      c = 1.0 - c;
    }
  }

  if (u_overlay > 0) {
    vec2 g = p / u_size * vec2(u_grid);
    float span = max(u_orange.y - u_orange.x, 1e-12);
    if (u_overlay == 1) {
      float n = (mapAt(ivec2(g)) - u_orange.x) / span;
      if (n >= u_thresh) c = mix(c, lut(n, u_ocmap), u_opacity);
    } else {
      // Bilinear between patch centres, then lines where the value crosses each tenth of the range.
      vec2 q = g - 0.5;
      ivec2 i0 = ivec2(floor(q));
      vec2 f = q - floor(q);
      float m = mix(mix(mapAt(i0), mapAt(i0 + ivec2(1, 0)), f.x), mix(mapAt(i0 + ivec2(0, 1)), mapAt(i0 + ivec2(1, 1)), f.x), f.y);
      float n = (m - u_orange.x) / span;
      float x = n * 10.0;
      float w = max(fwidth(x), 1e-6);
      float dpx = abs(fract(x + 0.5) - 0.5) / w;
      float line = 1.0 - smoothstep(0.55 * u_dpr, 1.25 * u_dpr, dpx);
      float level = floor(x + 0.5) / 10.0;
      if (level >= u_thresh && level > 0.0) c = mix(c, lut(level, u_ocmap), line * max(u_opacity, 0.5));
    }
  }
  outColor = vec4(c, 1.0);
}`;

interface Uploaded {
  img: WebGLTexture;
  maps: Record<MapSource, WebGLTexture | null>;
}

export interface DrawState {
  tile: TileData;
  display: Display;
  limits: Limits;
  mapRange: [number, number];
  centre: [number, number];
  /** CSS px per image px. */
  zoom: number;
  nan: [number, number, number];
}

export class Renderer {
  readonly gl: WebGL2RenderingContext;
  private prog: WebGLProgram;
  private loc: Record<string, WebGLUniformLocation | null> = {};
  private lut: WebGLTexture;
  private eq: WebGLTexture;
  private textures = new Map<number, Uploaded>();

  constructor(private canvas: HTMLCanvasElement) {
    const gl = canvas.getContext('webgl2', { antialias: false, premultipliedAlpha: true, alpha: true });
    if (!gl) throw new Error('This browser has no WebGL2, which the plate needs to draw float pixels');
    this.gl = gl;
    this.prog = this.link();
    gl.useProgram(this.prog);
    for (const name of FRAG.matchAll(/uniform \w+ \w+ (\w+);|uniform \w+ (\w+);/g)) {
      const n = name[1] ?? name[2];
      this.loc[n] = gl.getUniformLocation(this.prog, n);
    }
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    this.lut = this.texture();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, 256, 6, 0, gl.RGBA, gl.UNSIGNED_BYTE, lutTexels());
    this.eq = this.texture();
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, 256, 3, 0, gl.RED, gl.FLOAT, new Float32Array(256 * 3));
    gl.bindVertexArray(gl.createVertexArray());
  }

  private link(): WebGLProgram {
    const gl = this.gl;
    const shader = (type: number, src: string) => {
      const s = gl.createShader(type)!;
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(`Shader did not compile: ${gl.getShaderInfoLog(s)}`);
      return s;
    };
    const p = gl.createProgram()!;
    gl.attachShader(p, shader(gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, shader(gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(`Shader did not link: ${gl.getProgramInfoLog(p)}`);
    return p;
  }

  private texture(): WebGLTexture {
    const gl = this.gl;
    const t = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return t;
  }

  private mapTexture(values: Float32Array, cols: number, rows: number): WebGLTexture {
    const t = this.texture();
    this.gl.texImage2D(this.gl.TEXTURE_2D, 0, this.gl.R32F, cols, rows, 0, this.gl.RED, this.gl.FLOAT, values);
    return t;
  }

  has(index: number): boolean {
    return this.textures.has(index);
  }

  /** Sends a tile's pixels and maps to the GPU once, so drawing it later costs nothing. */
  upload(d: TileData, grid: [number, number]) {
    if (this.textures.has(d.index)) return;
    const gl = this.gl;
    const img = this.texture();
    if (d.planes && d.channels === 3) {
      const n = d.width * d.height;
      const rgb = new Float32Array(n * 3);
      const [r, g, b] = d.planes;
      for (let i = 0; i < n; i++) {
        rgb[i * 3] = r[i];
        rgb[i * 3 + 1] = g[i];
        rgb[i * 3 + 2] = b[i];
      }
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB32F, d.width, d.height, 0, gl.RGB, gl.FLOAT, rgb);
    } else if (d.planes) {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, d.width, d.height, 0, gl.RED, gl.FLOAT, d.planes[0]);
    } else {
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, d.width, d.height, 0, gl.RGBA, gl.UNSIGNED_BYTE, d.rgba);
    }
    const [rows, cols] = grid;
    let diff: Float32Array | null = null;
    if (d.map1) {
      diff = new Float32Array(d.map2.length);
      for (let i = 0; i < diff.length; i++) diff[i] = d.map2[i] - d.map1[i];
    }
    this.textures.set(d.index, {
      img,
      maps: {
        pass2: this.mapTexture(d.map2, cols, rows),
        pass1: d.map1 ? this.mapTexture(d.map1, cols, rows) : null,
        diff: diff ? this.mapTexture(diff, cols, rows) : null,
      },
    });
  }

  evict(index: number) {
    const u = this.textures.get(index);
    if (!u) return;
    this.gl.deleteTexture(u.img);
    for (const t of Object.values(u.maps)) if (t) this.gl.deleteTexture(t);
    this.textures.delete(index);
  }

  setEqualiser(values: Float32Array) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.eq);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 3, gl.RED, gl.FLOAT, values);
  }

  draw(s: DrawState, grid: [number, number]) {
    const gl = this.gl;
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(this.canvas.clientWidth * dpr));
    const h = Math.max(1, Math.round(this.canvas.clientHeight * dpr));
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
    gl.viewport(0, 0, w, h);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    const u = this.textures.get(s.tile.index);
    if (!u) return;
    const d = s.display;
    const mapTex = u.maps[d.source] ?? u.maps.pass2;

    const bind = (unit: number, tex: WebGLTexture, name: string) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(this.loc[name], unit);
    };
    bind(0, u.img, 'u_img');
    bind(1, this.lut, 'u_lut');
    bind(2, this.eq, 'u_eq');
    bind(3, mapTex!, 'u_map');

    const L = this.loc;
    gl.uniform2f(L.u_size, s.tile.width, s.tile.height);
    gl.uniform2f(L.u_canvas, w, h);
    gl.uniform2f(L.u_center, s.centre[0], s.centre[1]);
    gl.uniform1f(L.u_zoom, s.zoom * dpr);
    gl.uniform1f(L.u_dpr, dpr);
    gl.uniform1f(L.u_scale, s.tile.planes ? 1 : 255);
    gl.uniform1i(L.u_channels, s.tile.channels);
    const lim = (a: number[]) => [a[0], a[1] ?? a[0], a[2] ?? a[0]] as const;
    gl.uniform3f(L.u_lo, ...lim(s.limits.lo));
    gl.uniform3f(L.u_hi, ...lim(s.limits.hi));
    gl.uniform1i(L.u_mono, d.mono ? 1 : 0);
    gl.uniform1i(L.u_stretch, ['linear', 'sqrt', 'log', 'asinh', 'histeq'].indexOf(d.stretch));
    gl.uniform1i(L.u_cmap, cmapRow(d.cmap));
    gl.uniform1i(L.u_invert, d.invert ? 1 : 0);
    gl.uniform3f(L.u_nan, ...s.nan);
    gl.uniform1i(L.u_overlay, d.overlay ? (d.contours ? 2 : 1) : 0);
    gl.uniform1f(L.u_opacity, d.overlayOpacity);
    gl.uniform1i(L.u_ocmap, cmapRow(d.overlayCmap));
    gl.uniform2f(L.u_orange, s.mapRange[0], s.mapRange[1]);
    gl.uniform1f(L.u_thresh, d.threshold);
    gl.uniform2i(L.u_grid, grid[1], grid[0]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
}
