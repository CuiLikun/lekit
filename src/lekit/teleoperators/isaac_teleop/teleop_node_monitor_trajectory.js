function position3(value) {
  if (!Array.isArray(value) || value.length !== 3 || value.some(item => !Number.isFinite(item))) {
    throw new TypeError("position must contain three finite numbers");
  }
  return value.slice();
}

export class EngagementTrajectory {
  constructor({ maximumPoints = 240, minimumDistance = 0.004 } = {}) {
    if (!Number.isInteger(maximumPoints) || maximumPoints < 1) {
      throw new RangeError("maximumPoints must be a positive integer");
    }
    if (!Number.isFinite(minimumDistance) || minimumDistance < 0) {
      throw new RangeError("minimumDistance must be finite and non-negative");
    }
    this.maximumPoints = maximumPoints;
    this.minimumDistanceSquared = minimumDistance ** 2;
    this.points = [];
    this.wasEngaged = false;
  }

  update(engaged, position) {
    const active = Boolean(engaged);
    let changed = false;
    if (active && !this.wasEngaged) {
      this.points = [];
      changed = true;
    }
    if (active) {
      const next = position3(position);
      const latest = this.points.at(-1);
      const distanceSquared = latest == null
        ? Number.POSITIVE_INFINITY
        : next.reduce((total, value, index) => total + (value - latest[index]) ** 2, 0);
      if (distanceSquared >= this.minimumDistanceSquared) {
        this.points.push(next);
        if (this.points.length > this.maximumPoints) this.points.shift();
        changed = true;
      }
    }
    this.wasEngaged = active;
    return changed;
  }
}
