"""
Quality gate logic for backend (mirrors frontend implementation).
"""


class FallbackDecisionEngine:
    """Fallback decision engine matching frontend implementation."""

    def decide(self, metrics: dict) -> dict:
        """Make fallback mode decision based on quality metrics."""
        gate_results = {
            "segmentation": self._check_segmentation(metrics.get("mask", {})),
            "pose": self._check_pose(metrics.get("pose", {})),
            "track": self._check_track(metrics.get("track", {})),
            "depth": self._check_depth(metrics.get("depth", {})),
        }

        # Decision tree (Section E of PRD)
        if not gate_results["pose"]["passed"]:
            if gate_results["pose"].get("score", 0) < 0.3:
                mode = "render-only"
                bounds = {
                    "maxYaw": 0,
                    "maxPitch": 0,
                    "maxRoll": 0,
                    "maxTranslation": 0,
                    "maxTranslationDepthPercent": 0,
                    "clampToParallax": True,
                }
            else:
                mode = "micro-parallax"
                bounds = {
                    "maxYaw": 8,
                    "maxPitch": 4,
                    "maxRoll": 1,
                    "maxTranslation": 0.03,
                    "maxTranslationDepthPercent": 1,
                    "clampToParallax": True,
                }
        elif not gate_results["track"]["passed"]:
            mode = "2.5d-subject"
            bounds = {
                "maxYaw": 15,
                "maxPitch": 8,
                "maxRoll": 2,
                "maxTranslation": 0.05,
                "maxTranslationDepthPercent": 1.5,
                "clampToParallax": True,
            }
        else:
            mode = "full-orbit"
            bounds = {
                "maxYaw": 20,
                "maxPitch": 10,
                "maxRoll": 3,
                "maxTranslation": 0.1,
                "maxTranslationDepthPercent": 2,
                "clampToParallax": True,
            }

        return {
            "mode": mode,
            "bounds": bounds,
            "gate_results": gate_results,
        }

    def _check_segmentation(self, metrics: dict) -> dict:
        score = metrics.get("score", 0)
        passed = score >= 0.5
        pose_risk = metrics.get("high_coverage_frame_count", 0) / max(1, metrics.get("total_frames", 1)) > 0.3

        return {
            "passed": passed,
            "score": score,
            "poseRisk": pose_risk,
        }

    def _check_pose(self, metrics: dict) -> dict:
        inlier_ok = metrics.get("inlier_ratio", 0) >= 0.35
        reproj_ok = metrics.get("median_reprojection_error", 999) <= 2.0
        coverage_ok = metrics.get("good_coverage_frame_percent", 0) >= 0.6
        jitter_ok = metrics.get("jitter_score", 1) <= 0.5

        passed = inlier_ok and reproj_ok and coverage_ok and jitter_ok

        reasons = []
        if not inlier_ok:
            reasons.append(f"Inlier ratio {metrics.get('inlier_ratio', 0):.2f} below 0.35")
        if not reproj_ok:
            reasons.append(f"Reprojection error {metrics.get('median_reprojection_error', 0):.2f}px above 2.0px")
        if not coverage_ok:
            reasons.append(f"Coverage {metrics.get('good_coverage_frame_percent', 0):.1%} below 60%")
        if not jitter_ok:
            reasons.append(f"Jitter score {metrics.get('jitter_score', 0):.2f} above 0.5")

        return {
            "passed": passed,
            "score": metrics.get("score", 0),
            "reasons": reasons,
        }

    def _check_track(self, metrics: dict) -> dict:
        tracks_ok = metrics.get("num_tracks_kept", 0) >= 50
        lifespan_ok = metrics.get("median_lifespan", 0) >= 15
        conf_ok = metrics.get("median_confidence", 0) >= 0.6

        passed = tracks_ok and lifespan_ok and conf_ok

        reasons = []
        if not tracks_ok:
            reasons.append(f"Only {metrics.get('num_tracks_kept', 0)} tracks kept (need 50+)")
        if not lifespan_ok:
            reasons.append(f"Median lifespan {metrics.get('median_lifespan', 0)} frames (need 15+)")
        if not conf_ok:
            reasons.append(f"Median confidence {metrics.get('median_confidence', 0):.2f} (need 0.6+)")

        return {
            "passed": passed,
            "reasons": reasons,
        }

    def _check_depth(self, metrics: dict) -> dict:
        temporal_ok = metrics.get("temporal_consistency", 0) >= 0.7
        edge_ok = metrics.get("edge_stability", 0) >= 0.6

        passed = temporal_ok and edge_ok

        reasons = []
        if not temporal_ok:
            reasons.append(f"Temporal consistency {metrics.get('temporal_consistency', 0):.2f} below 0.7")
        if not edge_ok:
            reasons.append(f"Edge stability {metrics.get('edge_stability', 0):.2f} below 0.6")

        return {
            "passed": passed,
            "reasons": reasons,
        }
