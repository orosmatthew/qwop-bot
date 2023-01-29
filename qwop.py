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


class Character:

    def __init__(self, physics_space: pm.Space, leg_muscle_strength: float):
        self.leg_muscle_strength = leg_muscle_strength

        self.left_foot = PhysicsLimb(physics_space, group=1, width=30, height=20, mass=5, friction=0.6,
                                     position=(150, 300))

        self.left_leg = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=100, mass=10, friction=0.6,
                                              position=(150, 300))

        self.left_ankle = pm.PivotJoint(self.left_foot.body, self.left_leg.limb.body, (0, 25), (0, -45))
        physics_space.add(self.left_ankle)
        self.left_ankle_limit = pm.RotaryLimitJoint(self.left_foot.body, self.left_leg.limb.body, radians(-25),
                                                    radians(25))
        physics_space.add(self.left_ankle_limit)

        self.right_foot = PhysicsLimb(physics_space, group=1, width=30, height=20, mass=5, friction=0.6,
                                      position=(250, 300))

        self.right_leg = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=100, mass=10, friction=0.8,
                                               position=(250, 300))

        self.right_ankle = pm.PivotJoint(self.right_foot.body, self.right_leg.limb.body, (0, 25), (0, -45))
        physics_space.add(self.right_ankle)
        self.right_ankle_limit = pm.RotaryLimitJoint(self.right_foot.body, self.right_leg.limb.body, radians(-25),
                                                     radians(25))
        physics_space.add(self.right_ankle_limit)

        self.hip = pm.PivotJoint(self.left_leg.limb.body, self.right_leg.limb.body, (0, 50), (0, 50))
        physics_space.add(self.hip)
        self.hip_limit = pm.RotaryLimitJoint(self.left_leg.limb.body, self.right_leg.limb.body, radians(-120),
                                             radians(120))
        physics_space.add(self.hip_limit)

    def draw(self) -> None:
        self.left_foot.draw(rl.MAROON)
        self.right_foot.draw(rl.RED)
        self.left_leg.limb.draw(rl.GRAY)
        self.right_leg.limb.draw(rl.WHITE)

    def move_legs_q(self) -> None:
        self.left_leg.move_muscle(self.leg_muscle_strength, -radians(50))
        self.right_leg.move_muscle(self.leg_muscle_strength, radians(50))

    def move_legs_w(self) -> None:
        self.left_leg.move_muscle(self.leg_muscle_strength, radians(50))
        self.right_leg.move_muscle(self.leg_muscle_strength, -radians(50))

    def relax_legs(self) -> None:
        self.left_leg.relax_muscle()
        self.right_leg.relax_muscle()


def main():
    space = pm.Space()
    space.gravity = (0, -900.0)

    character = Character(space, leg_muscle_strength=1500000.0)

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

        camera.target = rl.Vector2(character.left_leg.limb.body.position.x, -character.left_leg.limb.body.position.y)

        rl.begin_drawing()
        rl.begin_mode_2d(camera)

        rl.clear_background(rl.BLACK)

        character.draw()

        rl.draw_rectangle_pro(rl.Rectangle(round(ground_body.position.x), round(-ground_body.position.y), 1000, 50),
                              rl.Vector2(1000 / 2, 50 / 2), 0.0, rl.GREEN)

        if rl.is_key_down(rl.KeyboardKey.KEY_Q):
            character.move_legs_q()
        elif rl.is_key_down(rl.KeyboardKey.KEY_W):
            character.move_legs_w()
        else:
            character.relax_legs()

        rl.end_mode_2d()
        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
