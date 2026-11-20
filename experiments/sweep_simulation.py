"""Synthetic occupancy-grid footprint checks. No robot access or execution."""
import math


def check_sweep(grid, pose, linear, angular, duration, radius, now, max_age=2.0):
    result = {'status': 'invalid', 'executable': False}
    try:
        values = [linear, angular, duration, radius, now, max_age, grid['resolution'],
                  grid['stamp'], pose[0], pose[1], pose[2], grid['origin'][0], grid['origin'][1]]
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in values):
            return result
        if not (0 <= linear <= .2 and abs(angular) <= .5 and 0 < duration <= 1
                and 0 < radius <= 1 and 0 < max_age <= 10 and 0 < grid['resolution'] <= 1):
            return result
        if grid['stamp'] > now or now - grid['stamp'] > max_age:
            result['status'] = 'stale'
            return result
        cells = grid['cells']
        if not cells or not cells[0] or len(cells) > 400 or len(cells[0]) > 400:
            return result
        height, width = len(cells), len(cells[0])
        if any(len(row) != width or any(type(cell) is not int or cell < -1 or cell > 100 for cell in row)
               for row in cells):
            return result
        res = grid['resolution']
        ox, oy = grid['origin']
        x0, y0, yaw = pose
        steps = int(math.ceil(duration / .05))
        dt = duration / steps
        inflated = radius + linear * dt / 2
        for i in range(steps + 1):
            t = i * dt
            if abs(angular) < 1e-9:
                x, y = x0 + linear * t * math.cos(yaw), y0 + linear * t * math.sin(yaw)
            else:
                x = x0 + linear / angular * (math.sin(yaw + angular * t) - math.sin(yaw))
                y = y0 - linear / angular * (math.cos(yaw + angular * t) - math.cos(yaw))
            c0, c1 = math.floor((x - inflated - ox) / res), math.floor((x + inflated - ox) / res)
            r0, r1 = math.floor((y - inflated - oy) / res), math.floor((y + inflated - oy) / res)
            for row in range(r0, r1 + 1):
                for col in range(c0, c1 + 1):
                    xmin, ymin = ox + col * res, oy + row * res
                    dx = max(xmin - x, 0, x - (xmin + res))
                    dy = max(ymin - y, 0, y - (ymin + res))
                    if dx * dx + dy * dy > inflated * inflated:
                        continue
                    if row < 0 or row >= height or col < 0 or col >= width or cells[row][col] == -1:
                        result['status'] = 'unknown'
                        return result
                    if cells[row][col] != 0:
                        result['status'] = 'blocked'
                        return result
        result['status'] = 'clear_in_simulation'
        return result
    except (KeyError, TypeError, ValueError, IndexError, OverflowError):
        return result
