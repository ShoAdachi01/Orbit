/**
 * Integration Tests for Orbit Snapshot Pipeline
 */

import { describe, it, expect } from 'vitest';
import { getModelFactory } from '../ml/ModelInterface';

describe('ML Model Stubs', () => {
  it('should create new pipeline model stubs', () => {
    const factory = getModelFactory();

    expect(factory.createSAM3DBody()).toBeDefined();
    expect(factory.createNanoBanana()).toBeDefined();
    expect(factory.createVeo()).toBeDefined();
  });
});

describe('End-to-End Simulation', () => {
  it('should simulate snapshot -> preview -> veo pipeline', async () => {
    const factory = getModelFactory();
    const sam3dBody = factory.createSAM3DBody();
    const nanoBanana = factory.createNanoBanana();
    const veo = factory.createVeo();

    await sam3dBody.initialize({});
    await nanoBanana.initialize({});
    await veo.initialize({});

    const snapshot = new Uint8Array(100 * 100 * 3);

    const reconstruction = await sam3dBody.reconstruct({
      image: snapshot,
      width: 100,
      height: 100,
    });

    expect(reconstruction.success).toBe(true);
    expect(reconstruction.data?.vertices.length).toBeGreaterThan(0);

    const preview = await nanoBanana.generate({
      image: snapshot,
      width: 100,
      height: 100,
      orientation: 'horizontal',
    });

    expect(preview.success).toBe(true);
    expect(preview.data?.image.length).toBe(snapshot.length);

    const veoResult = await veo.generate({
      referenceImage: preview.data!.image,
      width: 100,
      height: 100,
      durationSec: 3,
      orientation: 'horizontal',
      fps: 24,
    });

    expect(veoResult.success).toBe(true);
    expect(veoResult.data?.durationSec).toBe(3);

    sam3dBody.destroy();
    nanoBanana.destroy();
    veo.destroy();
  });
});
