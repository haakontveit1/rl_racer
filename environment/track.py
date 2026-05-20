import pygame
from pygame.math import Vector2
from config import *


# Default track: an L-shape. Make a new track by passing a different waypoint
# list to Track(...). The centerline, asphalt, start/finish lines, and car
# spawn position all derive from these waypoints.
# Named tracks. Pass `--track <name>` to play_model.py or main.py to select one.
# Add new tracks here by appending to the dict. The training script always uses
# DEFAULT_TRACK; change DEFAULT_TRACK below if you want to train on a different one.
TRACKS = {
    "simple_track": [(200, 150), (200, 600), (755, 600)],
    "chicane": [(150, 200), (150, 600), (500, 600), (500, 250), (850, 250), (850, 650)],
}
DEFAULT_TRACK = "chicane"
DEFAULT_WAYPOINTS = TRACKS[DEFAULT_TRACK]
DEFAULT_WIDTH = 200
START_OFFSET = 30  # how far behind the first waypoint the car spawns


class Track:
    def __init__(self, waypoints=None, width=DEFAULT_WIDTH):
        if waypoints is None:
            waypoints = DEFAULT_WAYPOINTS
        self.waypoints = [Vector2(*p) for p in waypoints]
        self.width = width
        self.half_width = width / 2
        self.half_width_sq = self.half_width ** 2

        # Cumulative arc length up to each waypoint — used for progress measurement.
        self._cumulative_length = [0.0]
        for i in range(1, len(self.waypoints)):
            seg_len = (self.waypoints[i] - self.waypoints[i - 1]).length()
            self._cumulative_length.append(self._cumulative_length[-1] + seg_len)
        self.total_length = self._cumulative_length[-1]

        # The centerline is just the waypoints themselves.
        self.centerline = self.waypoints

        # Perpendicular bars at the first and last waypoints, spanning full track width.
        self.start_line = self._perpendicular_bar(0)
        self.finish_line = self._perpendicular_bar(len(self.waypoints) - 1)

        # Car spawn: slightly behind the first waypoint, facing along first segment.
        first_dir = (self.waypoints[1] - self.waypoints[0]).normalize()
        self.start_position = self.waypoints[0] - first_dir * START_OFFSET
        self.start_angle = first_dir.as_polar()[1]

    def _perpendicular_bar(self, index):
        if index == 0:
            direction = (self.waypoints[1] - self.waypoints[0]).normalize()
        else:
            direction = (self.waypoints[index] - self.waypoints[index - 1]).normalize()
        perp = Vector2(-direction.y, direction.x)
        center = self.waypoints[index]
        return (center - perp * self.half_width, center + perp * self.half_width)

    def draw(self, surface):
        # Thick polyline + filled circles at waypoints = exact "distance ≤ half_width" shape.
        if len(self.waypoints) >= 2:
            pygame.draw.lines(surface, GRAY, False, self.waypoints, self.width)
        for wp in self.waypoints:
            pygame.draw.circle(surface, GRAY, (int(wp.x), int(wp.y)), int(self.half_width))

        pygame.draw.line(surface, WHITE, *self.start_line, 6)
        pygame.draw.line(surface, BLACK, *self.finish_line, 8)

    def is_on_track(self, point):
        return self._distance_squared_to_polyline(point) <= self.half_width_sq

    def crossed_finish(self, prev_position, curr_position):
        return self._segments_intersect(
            prev_position, curr_position, self.finish_line[0], self.finish_line[1]
        )

    def progress_along_centerline(self, point):
        best_distance_sq = float("inf")
        best_progress = 0.0
        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            ab = b - a
            ab_len_sq = ab.length_squared()
            if ab_len_sq == 0:
                continue
            t = max(0.0, min(1.0, (point - a).dot(ab) / ab_len_sq))
            projected = a + ab * t
            dist_sq = (point - projected).length_squared()
            if dist_sq < best_distance_sq:
                best_distance_sq = dist_sq
                best_progress = self._cumulative_length[i] + t * ab.length()
        return best_progress

    def signed_lateral_offset(self, point):
        # Perpendicular distance from centerline, signed by which side of the
        # segment direction the point lies on (via the 2D cross product).
        best_distance_sq = float("inf")
        best_offset = 0.0
        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            ab = b - a
            ab_len_sq = ab.length_squared()
            if ab_len_sq == 0:
                continue
            t = max(0.0, min(1.0, (point - a).dot(ab) / ab_len_sq))
            projected = a + ab * t
            diff = point - projected
            dist_sq = diff.length_squared()
            if dist_sq < best_distance_sq:
                best_distance_sq = dist_sq
                cross = ab.x * diff.y - ab.y * diff.x
                best_offset = cross / ab.length()
        return best_offset

    def cast_ray(self, origin, direction, max_distance, step=4):
        distance = 0
        while distance < max_distance:
            distance += step
            point = origin + direction * distance
            if not self.is_on_track(point):
                return distance
        return max_distance

    def _distance_squared_to_polyline(self, point):
        best_sq = float("inf")
        for i in range(len(self.waypoints) - 1):
            a = self.waypoints[i]
            b = self.waypoints[i + 1]
            ab = b - a
            ab_len_sq = ab.length_squared()
            if ab_len_sq == 0:
                continue
            t = max(0.0, min(1.0, (point - a).dot(ab) / ab_len_sq))
            projected = a + ab * t
            dist_sq = (point - projected).length_squared()
            if dist_sq < best_sq:
                best_sq = dist_sq
        return best_sq

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        def ccw(a, b, c):
            return (c.y - a.y) * (b.x - a.x) > (b.y - a.y) * (c.x - a.x)
        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)
