import pyray as rl
import pymunk as pm
from util import gen_rect_verts
from math import degrees, radians


class PhysicsLimb:
    def __init__(self, physics_space: pm.Space, group: int, width: float, height: float, mass: float, friction: float,
                 position: tuple[float, float]):
        # Save the width and height
        self._width = width
        self._height = height
        # Get the vertices for the limb
        self.verts = gen_rect_verts(width, height)
        # A moment is the rotational inertia of the limb.
        # This calculates the moment from the polygon
        self.moment = pm.moment_for_poly(mass, self.verts)
        # The body is the actual physics object
        self.body = pm.Body(mass, self.moment)
        # The shape describes the shape of the body
        self.shape = pm.Poly(self.body, self.verts)
        self.shape.friction = friction
        # This filter prevents other limbs from interacting.
        # Shapes with the same group number ignore each other
        self.shape.filter = pm.ShapeFilter(group)
        self.body.position = position

        # Add the limb to the physics world
        physics_space.add(self.body, self.shape)

    # Draw the limb with a specified color
    def draw(self, color: rl.Color) -> None:
        rl.draw_rectangle_pro(
            rl.Rectangle(round(self.body.position.x), round(-self.body.position.y), self._width, self._height),
            rl.Vector2(self._width / 2, self._height / 2),
            -degrees(self.body.angle), color)


class PhysicsLimbWithMuscle:
    def __init__(self, physics_space: pm.Space, group: int, width: float, height: float, mass: float, friction: float,
                 position: tuple[float, float]):
        # Create the base limb without the muscle
        self.limb = PhysicsLimb(physics_space, group, width, height, mass, friction, position)
        # Create muscle body which is a target for where the body should rotate to.
        # It is a kinematic body which means that it does not have physics but is
        # only moved using code that we write.
        self.muscle_body = pm.Body(body_type=pm.Body.KINEMATIC)
        # The muscle itself is a rotational spring between the muscle body target and the limb.
        # As we tell the muscle body to rotate, the spring will pull the limb to rotate the same way
        # with a springiness
        self.muscle = pm.DampedRotarySpring(self.muscle_body, self.limb.body, 0, 0.0, 6000.0)
        # Add it to the physics world
        physics_space.add(self.muscle_body, self.muscle)

    # Move the muscle with a specified strength and the angle relative to the limb
    def move_muscle(self, strength: float, angle: float) -> None:
        self.muscle.stiffness = strength
        self.muscle_body.angle = self.limb.body.angle + angle

    # Relax the muscle by making the rotational spring have no stiffness
    def relax_muscle(self) -> None:
        self.muscle.stiffness = 0.0


class Character:
    def __init__(self, physics_space: pm.Space, leg_muscle_strength: float, arm_muscle_strength: float):
        # Body parts
        self.torso = PhysicsLimb(physics_space, group=1, width=40, height=150, mass=0.5, friction=1,
                                 position=(200, 600))

        self.neck = PhysicsLimbWithMuscle(physics_space, group=1, width=20, height=10, mass=0.05, friction=1,
                                          position=(230, 630))

        self.head = PhysicsLimb(physics_space, group=1, width=40, height=40, mass=0.3, friction=1,
                                position=(230, 650))

        self.right_biceps = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=80, mass=0.1, friction=0.6,
                                                  position=(230, 400))
        self.left_biceps = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=80, mass=0.1, friction=0.6,
                                                 position=(230, 400))

        self.right_forearm = PhysicsLimb(physics_space, group=1, width=15, height=70, mass=0.1, friction=0.6,
                                         position=(230, 400))
        self.left_forearm = PhysicsLimb(physics_space, group=1, width=15, height=70, mass=0.1, friction=0.6,
                                        position=(230, 400))

        # Connection (Joints)
        self.neck_to_torso = pm.PivotJoint(self.torso.body, self.neck.limb.body, (0, 75), (
            0, -5))  # The anchor point on body A, the anchor point on body B; for the last two variables

        self.neck_to_head = pm.PivotJoint(self.head.body, self.neck.limb.body, (0, -20), (0, 5))

        self.right_biceps_to_forearm = pm.PivotJoint(self.right_biceps.limb.body, self.right_forearm.body, (0, -40),
                                                     (0, 35))

        self.left_biceps_to_forearm = pm.PivotJoint(self.left_biceps.limb.body, self.left_forearm.body, (0, -40),
                                                    (0, 35))

        self.right_arm_to_torso = pm.PivotJoint(self.right_biceps.limb.body, self.torso.body, (3, 45), (0, 66))

        self.left_arm_to_torso = pm.PivotJoint(self.left_biceps.limb.body, self.torso.body, (-3, 45), (0, 66))

        physics_space.add(self.right_biceps_to_forearm)
        physics_space.add(self.left_biceps_to_forearm)

        physics_space.add(self.neck_to_torso)
        physics_space.add(self.neck_to_head)

        physics_space.add(self.right_arm_to_torso)
        physics_space.add(self.left_arm_to_torso)

        # Limits for the elbows, shoulders, neck (to body and to head)
        self.right_elbow_limit = pm.RotaryLimitJoint(self.right_biceps.limb.body, self.right_forearm.body, radians(20),
                                                     radians(155))

        self.left_elbow_limit = pm.RotaryLimitJoint(self.left_biceps.limb.body, self.left_forearm.body, radians(20),
                                                    radians(155))

        self.right_shoulder_limit = pm.RotaryLimitJoint(self.right_biceps.limb.body, self.torso.body, radians(-45),
                                                        radians(70))

        self.left_shoulder_limit = pm.RotaryLimitJoint(self.left_biceps.limb.body, self.torso.body, radians(-45),
                                                       radians(70))

        self.neck_to_torso_limit = pm.RotaryLimitJoint(self.neck.limb.body, self.torso.body, radians(-50),
                                                       radians(50))

        self.neck_to_head_limit = pm.RotaryLimitJoint(self.neck.limb.body, self.head.body, radians(-10),
                                                      radians(10))

        physics_space.add(self.right_elbow_limit)
        physics_space.add(self.left_elbow_limit)
        physics_space.add(self.right_shoulder_limit)
        physics_space.add(self.left_shoulder_limit)
        physics_space.add(self.neck_to_torso_limit)
        physics_space.add(self.neck_to_head_limit)

        self.leg_muscle_strength = leg_muscle_strength
        self.arm_muscle_strength = arm_muscle_strength

        self.left_foot = PhysicsLimb(physics_space, group=1, width=30, height=20, mass=1, friction=0.6,
                                     position=(50, 300))

        self.left_calf = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=100, mass=3, friction=0.6,
                                               position=(100, 300))

        self.left_ankle = pm.PivotJoint(self.left_foot.body, self.left_calf.limb.body, (0, 25), (0, -45))
        physics_space.add(self.left_ankle)

        self.left_ankle_limit = pm.RotaryLimitJoint(self.left_foot.body, self.left_calf.limb.body, radians(-25),
                                                    radians(25))
        physics_space.add(self.left_ankle_limit)

        self.left_leg = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=100, mass=3, friction=0.6,
                                              position=(100, 300))

        self.left_knee = pm.PivotJoint(self.left_calf.limb.body, self.left_leg.limb.body, (0, 50), (0, -50))
        physics_space.add(self.left_knee)
        self.left_knee_limit = pm.RotaryLimitJoint(self.left_calf.limb.body, self.left_leg.limb.body, radians(0),
                                                   radians(140))
        physics_space.add(self.left_knee_limit)

        self.right_foot = PhysicsLimb(physics_space, group=1, width=30, height=20, mass=1, friction=0.6,
                                      position=(250, 300))

        self.right_calf = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=100, mass=3, friction=0.6,
                                                position=(250, 300))

        self.right_ankle = pm.PivotJoint(self.right_foot.body, self.right_calf.limb.body, (0, 25), (0, -45))
        physics_space.add(self.right_ankle)
        self.right_ankle_limit = pm.RotaryLimitJoint(self.right_foot.body, self.right_calf.limb.body, radians(-25),
                                                     radians(25))
        physics_space.add(self.right_ankle_limit)

        self.right_leg = PhysicsLimbWithMuscle(physics_space, group=1, width=15, height=100, mass=3, friction=0.8,
                                               position=(250, 300))

        self.right_knee = pm.PivotJoint(self.right_calf.limb.body, self.right_leg.limb.body, (0, 50), (0, -50))
        physics_space.add(self.right_knee)
        self.right_knee_limit = pm.RotaryLimitJoint(self.right_calf.limb.body, self.right_leg.limb.body, radians(0),
                                                    radians(140))
        physics_space.add(self.right_knee_limit)

        self.hip_right = pm.PivotJoint(self.right_leg.limb.body, self.torso.body, (0, 50), (-5, -75))
        self.hip_left = pm.PivotJoint(self.left_leg.limb.body, self.torso.body, (0, 50), (5, -75))

        physics_space.add(self.hip_right)
        physics_space.add(self.hip_left)

        self.hip_limit = pm.RotaryLimitJoint(self.left_leg.limb.body, self.right_leg.limb.body, radians(-120),
                                             radians(120))

        self.body_left_leg_limit = pm.RotaryLimitJoint(self.left_leg.limb.body, self.torso.body, radians(-95),
                                                       radians(40))
        self.body_right_leg_limit = pm.RotaryLimitJoint(self.right_leg.limb.body, self.torso.body, radians(-95),
                                                        radians(40))

        physics_space.add(self.hip_limit)
        physics_space.add(self.body_left_leg_limit)
        physics_space.add(self.body_right_leg_limit)

    def draw(self) -> None:
        self.left_leg.limb.draw(rl.GRAY)
        self.right_leg.limb.draw(rl.WHITE)

        self.left_calf.limb.draw(rl.GRAY)
        self.right_calf.limb.draw(rl.WHITE)

        self.left_foot.draw(rl.MAROON)
        self.right_foot.draw(rl.RED)

        self.torso.draw(rl.MAROON)
        self.head.draw(rl.BROWN)

        self.neck.limb.draw(rl.BROWN)

        self.left_biceps.limb.draw(rl.GRAY)
        self.right_biceps.limb.draw(rl.WHITE)
        self.left_forearm.draw(rl.GRAY)
        self.right_forearm.draw(rl.WHITE)

    def move_legs_q(self) -> None:
        self.left_leg.move_muscle(self.leg_muscle_strength, -radians(40))
        self.right_leg.move_muscle(self.leg_muscle_strength, radians(40))
        self.right_biceps.move_muscle(self.arm_muscle_strength, radians(120))
        self.left_biceps.move_muscle(self.arm_muscle_strength, radians(-120))

    def move_legs_w(self) -> None:
        self.left_leg.move_muscle(self.leg_muscle_strength, radians(40))
        self.right_leg.move_muscle(self.leg_muscle_strength, -radians(40))
        self.right_biceps.move_muscle(self.arm_muscle_strength, radians(-120))
        self.left_biceps.move_muscle(self.arm_muscle_strength, radians(120))

    def move_knees_o(self) -> None:
        self.left_calf.move_muscle(self.leg_muscle_strength, radians(40))
        self.right_calf.move_muscle(self.leg_muscle_strength, -radians(40))

    def move_knees_p(self) -> None:
        self.left_calf.move_muscle(self.leg_muscle_strength, -radians(40))
        self.right_calf.move_muscle(self.leg_muscle_strength, radians(40))

    def hold_legs(self) -> None:
        self.left_leg.move_muscle(self.leg_muscle_strength, 0)
        self.right_leg.move_muscle(self.leg_muscle_strength, 0)

    def hold_knees(self) -> None:
        self.left_calf.move_muscle(self.leg_muscle_strength, 0)
        self.right_calf.move_muscle(self.leg_muscle_strength, 0)

    def relax_legs(self) -> None:
        self.left_leg.relax_muscle()
        self.right_leg.relax_muscle()

    def relax_knees(self) -> None:
        self.left_calf.relax_muscle()
        self.right_calf.relax_muscle()
