/**
 * MP4 Export Pipeline
 * Converts rendered frames to MP4 video using FFmpeg WASM
 */

import { RenderFrame } from '../render/RenderPipeline';

export interface ExportConfig {
  /** Output video codec */
  codec: 'h264' | 'h265';
  /** Video quality (CRF value, lower = better, 18-28 typical) */
  quality: number;
  /** Output bitrate in kbps (optional, overrides quality) */
  bitrate?: number;
  /** Pixel format */
  pixelFormat: 'yuv420p' | 'yuv444p';
  /** Include audio from source */
  includeAudio: boolean;
}

const DEFAULT_CONFIG: ExportConfig = {
  codec: 'h264',
  quality: 23,
  pixelFormat: 'yuv420p',
  includeAudio: false,
};

export interface ExportResult {
  data: Uint8Array;
  duration: number;
  frameCount: number;
  size: number;
}

export class MP4Export {
  private config: ExportConfig;
  private ffmpeg: any = null;

  constructor(config: Partial<ExportConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Initialize FFmpeg WASM
   */
  async initialize(): Promise<void> {
    // Dynamic import of FFmpeg
    const { FFmpeg } = await import('@ffmpeg/ffmpeg');
    const { fetchFile } = await import('@ffmpeg/util');

    this.ffmpeg = new FFmpeg();

    // Load FFmpeg core
    await this.ffmpeg.load();

    // Store fetchFile for later use
    (this as any).fetchFile = fetchFile;
  }

  /**
   * Export frames to MP4
   */
  async export(
    frames: RenderFrame[],
    fps: number,
    onProgress?: (progress: number) => void
  ): Promise<ExportResult> {
    if (!this.ffmpeg) {
      throw new Error('FFmpeg not initialized. Call initialize() first.');
    }

    const { fetchFile } = this as any;

    // Write frames as images
    onProgress?.(0);

    for (let i = 0; i < frames.length; i++) {
      const frame = frames[i];
      const pngData = this.frameToPNG(frame);

      await this.ffmpeg.writeFile(
        `frame_${i.toString().padStart(6, '0')}.png`,
        pngData
      );

      onProgress?.((i / frames.length) * 50);
    }

    // Build FFmpeg command
    const outputFile = 'output.mp4';
    const args = this.buildFFmpegArgs(fps, frames.length, outputFile);

    // Run FFmpeg
    onProgress?.(50);
    await this.ffmpeg.exec(args);
    onProgress?.(90);

    // Read output
    const data = await this.ffmpeg.readFile(outputFile);

    // Cleanup temp files
    for (let i = 0; i < frames.length; i++) {
      try {
        await this.ffmpeg.deleteFile(`frame_${i.toString().padStart(6, '0')}.png`);
      } catch {
        // Ignore cleanup errors
      }
    }
    try {
      await this.ffmpeg.deleteFile(outputFile);
    } catch {
      // Ignore cleanup errors
    }

    onProgress?.(100);

    return {
      data: data as Uint8Array,
      duration: frames.length / fps,
      frameCount: frames.length,
      size: (data as Uint8Array).length,
    };
  }

  /**
   * Convert RGBA frame data to PNG
   */
  private frameToPNG(frame: RenderFrame): Uint8Array {
    // Create a simple PNG encoder
    // In production, use a proper PNG library

    const { data, width, height } = frame;

    // Create canvas and draw frame
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d')!;

    const imageData = new ImageData(
      new Uint8ClampedArray(data),
      width,
      height
    );
    ctx.putImageData(imageData, 0, 0);

    // Convert to PNG blob synchronously (simplified)
    // In practice, use canvas.convertToBlob() or a PNG encoder
    return this.encodeSimplePNG(data, width, height);
  }

  /**
   * Simple PNG encoder (placeholder - use a real library in production)
   */
  private encodeSimplePNG(rgba: Uint8Array, width: number, height: number): Uint8Array {
    // PNG signature
    const signature = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);

    // IHDR chunk
    const ihdr = this.createIHDRChunk(width, height);

    // IDAT chunk (uncompressed for simplicity)
    const idat = this.createIDATChunk(rgba, width, height);

    // IEND chunk
    const iend = this.createIENDChunk();

    // Concatenate all chunks
    const totalLength = signature.length + ihdr.length + idat.length + iend.length;
    const png = new Uint8Array(totalLength);

    let offset = 0;
    png.set(signature, offset); offset += signature.length;
    png.set(ihdr, offset); offset += ihdr.length;
    png.set(idat, offset); offset += idat.length;
    png.set(iend, offset);

    return png;
  }

  private createIHDRChunk(width: number, height: number): Uint8Array {
    const data = new Uint8Array(13);
    const view = new DataView(data.buffer);

    view.setUint32(0, width, false);
    view.setUint32(4, height, false);
    data[8] = 8;  // bit depth
    data[9] = 6;  // color type (RGBA)
    data[10] = 0; // compression
    data[11] = 0; // filter
    data[12] = 0; // interlace

    return this.createChunk('IHDR', data);
  }

  private createIDATChunk(rgba: Uint8Array, width: number, height: number): Uint8Array {
    // Add filter byte (0 = no filter) to each row
    const filteredData = new Uint8Array(height * (1 + width * 4));

    for (let y = 0; y < height; y++) {
      filteredData[y * (1 + width * 4)] = 0; // Filter byte
      for (let x = 0; x < width * 4; x++) {
        filteredData[y * (1 + width * 4) + 1 + x] = rgba[y * width * 4 + x];
      }
    }

    // Compress with deflate (using simple zlib wrapper)
    const compressed = this.deflate(filteredData);

    return this.createChunk('IDAT', compressed);
  }

  private createIENDChunk(): Uint8Array {
    return this.createChunk('IEND', new Uint8Array(0));
  }

  private createChunk(type: string, data: Uint8Array): Uint8Array {
    const chunk = new Uint8Array(12 + data.length);
    const view = new DataView(chunk.buffer);

    // Length
    view.setUint32(0, data.length, false);

    // Type
    for (let i = 0; i < 4; i++) {
      chunk[4 + i] = type.charCodeAt(i);
    }

    // Data
    chunk.set(data, 8);

    // CRC32
    const crcData = new Uint8Array(4 + data.length);
    crcData.set(chunk.subarray(4, 8), 0);
    crcData.set(data, 4);
    const crc = this.crc32(crcData);
    view.setUint32(8 + data.length, crc, false);

    return chunk;
  }

  private deflate(data: Uint8Array): Uint8Array {
    // Simple uncompressed deflate block
    // In production, use pako or similar

    // Zlib header
    const header = new Uint8Array([0x78, 0x01]);

    // Uncompressed blocks
    const blocks: Uint8Array[] = [];
    const maxBlockSize = 65535;

    for (let i = 0; i < data.length; i += maxBlockSize) {
      const remaining = data.length - i;
      const blockSize = Math.min(maxBlockSize, remaining);
      const isLast = i + blockSize >= data.length;

      const blockHeader = new Uint8Array(5);
      blockHeader[0] = isLast ? 0x01 : 0x00;
      blockHeader[1] = blockSize & 0xff;
      blockHeader[2] = (blockSize >> 8) & 0xff;
      blockHeader[3] = ~blockSize & 0xff;
      blockHeader[4] = (~blockSize >> 8) & 0xff;

      blocks.push(blockHeader);
      blocks.push(data.subarray(i, i + blockSize));
    }

    // Adler32 checksum
    const adler = this.adler32(data);
    const checksum = new Uint8Array(4);
    const checksumView = new DataView(checksum.buffer);
    checksumView.setUint32(0, adler, false);

    // Concatenate
    const totalLength = header.length +
      blocks.reduce((sum, b) => sum + b.length, 0) +
      checksum.length;

    const result = new Uint8Array(totalLength);
    let offset = 0;

    result.set(header, offset); offset += header.length;
    for (const block of blocks) {
      result.set(block, offset); offset += block.length;
    }
    result.set(checksum, offset);

    return result;
  }

  private crc32(data: Uint8Array): number {
    let crc = 0xffffffff;

    for (let i = 0; i < data.length; i++) {
      crc ^= data[i];
      for (let j = 0; j < 8; j++) {
        crc = (crc >>> 1) ^ (crc & 1 ? 0xedb88320 : 0);
      }
    }

    return crc ^ 0xffffffff;
  }

  private adler32(data: Uint8Array): number {
    let a = 1;
    let b = 0;

    for (let i = 0; i < data.length; i++) {
      a = (a + data[i]) % 65521;
      b = (b + a) % 65521;
    }

    return (b << 16) | a;
  }

  /**
   * Build FFmpeg command arguments
   */
  private buildFFmpegArgs(fps: number, frameCount: number, output: string): string[] {
    const args = [
      '-framerate', fps.toString(),
      '-i', 'frame_%06d.png',
    ];

    // Codec settings
    if (this.config.codec === 'h264') {
      args.push('-c:v', 'libx264');
    } else {
      args.push('-c:v', 'libx265');
    }

    // Quality/bitrate
    if (this.config.bitrate) {
      args.push('-b:v', `${this.config.bitrate}k`);
    } else {
      args.push('-crf', this.config.quality.toString());
    }

    // Pixel format
    args.push('-pix_fmt', this.config.pixelFormat);

    // Preset for speed/quality tradeoff
    args.push('-preset', 'medium');

    // Output
    args.push('-y', output);

    return args;
  }

  destroy(): void {
    this.ffmpeg = null;
  }
}

/**
 * Utility to download blob as file
 */
export function downloadBlob(data: Uint8Array, filename: string): void {
  // Create a new ArrayBuffer copy to ensure proper typing for Blob
  const buffer = new Uint8Array(data).buffer as ArrayBuffer;
  const blob = new Blob([buffer], { type: 'video/mp4' });
  const url = URL.createObjectURL(blob);

  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();

  URL.revokeObjectURL(url);
}
