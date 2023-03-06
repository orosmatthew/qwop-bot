import pyray as rl
import pymunk as pm
import numpy as np
from character import Character
from neural_network import NeuralNetwork


# TODO: This class is a mess, need to refactor. Could really just be a function
class CharacterData:
    def __init__(self, torso_position: tuple,
                 head_position: tuple,
                 right_forearm_position: tuple,
                 right_biceps_position: tuple,
                 left_forearm_position: tuple,
                 left_biceps_position: tuple,
                 right_leg_position: tuple,
                 right_calf_position: tuple,
                 right_foot_position: tuple,
                 left_leg_position: tuple,
                 left_calf_position: tuple,
                 left_foot_position: tuple,
                 ):
        self.torso_position = torso_position
        self.head_position = head_position
        self.right_forearm_position = right_forearm_position
        self.right_biceps_position = right_biceps_position
        self.left_forearm_position = left_forearm_position
        self.left_biceps_position = left_biceps_position
        self.right_leg_position = right_leg_position
        self.right_calf_position = right_calf_position
        self.right_foot_position = right_foot_position
        self.left_leg_position = left_leg_position
        self.left_calf_position = left_calf_position
        self.left_foot_position = left_foot_position

    def get_position_all(self):
        return (self.torso_position,
                self.head_position,
                self.right_forearm_position,
                self.right_biceps_position,
                self.left_forearm_position,
                self.left_biceps_position,
                self.right_leg_position,
                self.right_calf_position,
                self.right_foot_position,
                self.left_leg_position,
                self.left_calf_position,
                self.left_foot_position)

    def get_position_all_x(self):
        return [item[0] for item in self.get_position_all()]

    def get_position_all_y(self):
        return [item[1] for item in self.get_position_all()]


def main():
    space = pm.Space()
    space.gravity = (0, -900.0)

    character = Character(space, leg_muscle_strength=1_000_000.0, arm_muscle_strength=50_000.0)

    ground_body = pm.Body(body_type=pm.Body.STATIC)
    ground_body.position = 300, 150
    ground_poly = [
        (-500, -25),
        (-500, 25),
        (500, 25),
        (500, -25),
    ]
    ground_shape = pm.Poly(ground_body, ground_poly)
    ground_shape.friction = 0.8
    ground_shape.collision_type = pm.Body.STATIC
    space.add(ground_body, ground_shape)

    rl.set_target_fps(60)

    camera = rl.Camera2D(rl.Vector2(1280 / 2, 720 / 2), rl.Vector2(0, 0), 0.0, 1.0)
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)

    neural_network = NeuralNetwork()

    rl.init_window(1280, 720, "QWOP-BOT")
    while not rl.window_should_close():
        space.step(1.0 / 60.0)

        camera.target = rl.Vector2(character.torso.body.position.x, -character.torso.body.position.y + 100)

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
            character.hold_legs()

        if rl.is_key_down(rl.KeyboardKey.KEY_O):
            character.move_knees_o()
        elif rl.is_key_down(rl.KeyboardKey.KEY_P):
            character.move_knees_p()
        else:
            character.hold_knees()

        rl.end_mode_2d()

        positions = CharacterData(character.torso.body.position,
                                  character.head.body.position,
                                  character.right_forearm.body.position,
                                  character.right_biceps.limb.body.position,
                                  character.left_forearm.body.position,
                                  character.left_biceps.limb.body.position,
                                  character.right_leg.limb.body.position,
                                  character.right_calf.limb.body.position,
                                  character.right_foot.body.position,
                                  character.left_leg.limb.body.position,
                                  character.left_calf.limb.body.position,
                                  character.left_foot.body.position)

        inputs = np.asarray(positions.get_position_all_x() + positions.get_position_all_y())

        output = neural_network.feedforward(inputs)

        def on_collision(arbiter, space, data):
            # Get the shapes that collided
            shape_1, shape_2 = arbiter.shapes
            list_of_shapes = [character.head.shape,
                              character.torso.shape,
                              character.right_biceps.limb.shape,
                              character.right_forearm.shape,
                              character.left_biceps.limb.shape,
                              character.left_forearm.shape]

            # Check if the colliding shapes belong to the head and floor
            if (shape_1 in list_of_shapes and shape_2 == ground_shape) or (
                    shape_1 == ground_shape and shape_2 in list_of_shapes):
                print("UPPER-BODY TOUCHED THE FLOOR")

        # # Add the collision handler to the space
        # handler = space.add_collision_handler(character.head.shape.collision_type, ground_shape.collision_type)
        # handler.begin = on_collision

        rl.draw_text("Distance: " + str(round(positions.left_foot_position[0], 0) / 1000.0) + "m", 20, 0, 50,
                     rl.Color(153, 204, 255, 255))
        rl.draw_text("Time: " + str(round(rl.get_time(), 2)), 20, 50, 50, rl.Color(153, 204, 255, 255))
        # break
        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
