import pyray as rl
import pymunk as pm
from math import degrees, radians


def gen_rect_verts(width: float, height: float) -> list[tuple[float, float]]:
    return [
        (- width / 2, -height / 2),
        (- width / 2, height / 2),
        (width / 2, height / 2),
        (width / 2, -height / 2),
    ]


class PhysicsLimb:
    def __init__(self, physics_space: pm.Space, group: int, width: float, height: float, mass: float, friction: float,
                 position: tuple[float, float]):
        self._width = width
        self._height = height

        self.verts = gen_rect_verts(width, height)
        self.moment = pm.moment_for_poly(mass, self.verts)
        self.body = pm.Body(mass, self.moment)
        self.shape = pm.Poly(self.body, self.verts)
        self.shape.friction = friction
        self.shape.filter = pm.ShapeFilter(group)
        self.body.position = position

        physics_space.add(self.body, self.shape)

    def draw(self, color: rl.Color) -> None:
        rl.draw_rectangle_pro(
            rl.Rectangle(round(self.body.position.x), round(-self.body.position.y), self._width, self._height),
            rl.Vector2(self._width / 2, self._height / 2),
            -degrees(self.body.angle), color)


class PhysicsLimbWithMuscle:
    def __init__(self, physics_space: pm.Space, group: int, width: float, height: float, mass: float, friction: float,
                 position: tuple[float, float]):
        self.limb = PhysicsLimb(physics_space, group, width, height, mass, friction, position)

        self.muscle_body = pm.Body(body_type=pm.Body.KINEMATIC)
        self.muscle = pm.DampedRotarySpring(self.muscle_body, self.limb.body, 0, 0.0, 6000.0)
        physics_space.add(self.muscle_body, self.muscle)

    def move_muscle(self, strength: float, angle: float) -> None:
        self.muscle.stiffness = strength
        self.muscle_body.angle = self.limb.body.angle + angle

    def relax_muscle(self) -> None:
        self.muscle.stiffness = 0.0


space = pm.Space()
space.gravity = (0, -900.0)

left_foot = PhysicsLimb(space, group=1, width=30, height=20, mass=5, friction=0.6, position=(150, 300))

left_leg = PhysicsLimbWithMuscle(space, group=1, width=15, height=100, mass=10, friction=0.6, position=(150, 300))

pivot_left_ankle = pm.PivotJoint(left_foot.body, left_leg.limb.body, (0, 25), (0, -45))
space.add(pivot_left_ankle)
rotary_limit_left_ankle = pm.RotaryLimitJoint(left_foot.body, left_leg.limb.body, radians(-25), radians(25))
space.add(rotary_limit_left_ankle)

right_foot = PhysicsLimb(space, group=1, width=30, height=20, mass=5, friction=0.6, position=(250, 300))

right_leg = PhysicsLimbWithMuscle(space, group=1, width=15, height=100, mass=10, friction=0.8, position=(250, 300))

pivot_right_ankle = pm.PivotJoint(right_foot.body, right_leg.limb.body, (0, 25), (0, -45))
space.add(pivot_right_ankle)
rotary_limit_right_ankle = pm.RotaryLimitJoint(right_foot.body, right_leg.limb.body, radians(-25), radians(25))
space.add(rotary_limit_right_ankle)

pivot_hip = pm.PivotJoint(left_leg.limb.body, right_leg.limb.body, (0, 50), (0, 50))
space.add(pivot_hip)
rotary_limit_hip = pm.RotaryLimitJoint(left_leg.limb.body, right_leg.limb.body, radians(-120), radians(120))
space.add(rotary_limit_hip)

ground_body = pm.Body(body_type=pm.Body.STATIC)
ground_body.position = 500, 0
ground_poly = [
    (-500, -25),
    (-500, 25),
    (500, 25),
    (500, -25),
]
ground_shape = pm.Poly(ground_body, ground_poly)
ground_shape.friction = 0.8
space.add(ground_body, ground_shape)

rl.set_target_fps(60)

camera = rl.Camera2D(rl.Vector2(1280 / 2, 720 / 2), rl.Vector2(0, 0), 0.0, 1.0)
rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)

rl.init_window(1280, 720, "QWOP-BOT")
while not rl.window_should_close():
    space.step(1.0 / 60.0)

    camera.target = rl.Vector2(left_leg.limb.body.position.x, -left_leg.limb.body.position.y)

    rl.begin_drawing()
    rl.begin_mode_2d(camera)

    rl.clear_background(rl.BLACK)

    left_foot.draw(rl.MAROON)
    right_foot.draw(rl.RED)
    left_leg.limb.draw(rl.GRAY)
    right_leg.limb.draw(rl.WHITE)

    rl.draw_rectangle_pro(rl.Rectangle(round(ground_body.position.x), round(-ground_body.position.y), 1000, 50),
                          rl.Vector2(1000 / 2, 50 / 2), 0.0, rl.GREEN)

    leg_muscle_strength = 1500000.0
    if rl.is_key_down(rl.KeyboardKey.KEY_Q):
        left_leg.move_muscle(leg_muscle_strength, -radians(50))
        right_leg.move_muscle(leg_muscle_strength, radians(50))
    elif rl.is_key_down(rl.KeyboardKey.KEY_W):
        left_leg.move_muscle(leg_muscle_strength, radians(50))
        right_leg.move_muscle(leg_muscle_strength, -radians(50))
    else:
        left_leg.relax_muscle()
        right_leg.relax_muscle()

    rl.end_mode_2d()
    rl.end_drawing()
rl.close_window()
