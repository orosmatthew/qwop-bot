import pyray as rl
import pymunk as pm
import numpy as np
from character import Character
from neural_network import NeuralNetwork
from util import vec2d_to_arr


def character_data_list(character: Character) -> list[float]:
    data: list[float] = []
    data.extend(vec2d_to_arr(character.torso.body.position))
    data.extend(vec2d_to_arr(character.head.body.position))
    data.extend(vec2d_to_arr(character.right_forearm.body.position))
    data.extend(vec2d_to_arr(character.right_biceps.limb.body.position))
    data.extend(vec2d_to_arr(character.left_forearm.body.position))
    data.extend(vec2d_to_arr(character.left_biceps.limb.body.position))
    data.extend(vec2d_to_arr(character.right_leg.limb.body.position))
    data.extend(vec2d_to_arr(character.right_calf.limb.body.position))
    data.extend(vec2d_to_arr(character.right_foot.body.position))
    data.extend(vec2d_to_arr(character.left_leg.limb.body.position))
    data.extend(vec2d_to_arr(character.left_calf.limb.body.position))
    data.extend(vec2d_to_arr(character.left_foot.body.position))
    return data


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

        inputs = np.asarray(character_data_list(character))

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


        rl.draw_text("Distance: " + str(round(character.torso.body.position.x, 0) / 1000.0) + "m", 20, 0, 50,
                     rl.Color(153, 204, 255, 255))
        rl.draw_text("Time: " + str(round(rl.get_time(), 2)), 20, 50, 50, rl.Color(153, 204, 255, 255))

        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
