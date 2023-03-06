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


class CharacterSimulation:
    def __init__(self, ground_body: pm.Body, ground_shape: pm.Shape):
        self.space: pm.Space = pm.Space()
        self.space.gravity = (0, -900.0)

        self.character: Character = Character(self.space, leg_muscle_strength=1_000_000.0, arm_muscle_strength=50_000.0)

        self.space.add(ground_body, ground_shape)

        self.neural_network: NeuralNetwork = NeuralNetwork()

    def step(self) -> None:
        self.space.step(1.0 / 60.0)

    def sim_position(self) -> rl.Vector2:
        return rl.Vector2(self.character.torso.body.position.x, self.character.torso.body.position.y)

    def draw_position(self) -> rl.Vector2:
        return rl.Vector2(self.character.torso.body.position.x, -self.character.torso.body.position.y + 100)

    def draw_character(self) -> None:
        self.character.draw()

    def character_move_legs_q(self) -> None:
        self.character.move_legs_q()

    def character_move_legs_w(self) -> None:
        self.character.move_legs_w()

    def character_move_knees_o(self) -> None:
        self.character.move_knees_o()

    def character_move_knees_p(self) -> None:
        self.character.move_knees_p()


def main():
    rl.set_target_fps(60)
    camera = rl.Camera2D(rl.Vector2(1280 / 2, 720 / 2), rl.Vector2(0, 0), 0.0, 1.0)
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
    rl.init_window(1280, 720, "QWOP-BOT")

    ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
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

    sim = CharacterSimulation(ground_body, ground_shape)

    while not rl.window_should_close():

        sim.step()

        camera.target = sim.draw_position()

        rl.begin_drawing()
        rl.begin_mode_2d(camera)

        rl.clear_background(rl.BLACK)

        sim.draw_character()

        rl.draw_rectangle_pro(rl.Rectangle(round(ground_body.position.x), round(-ground_body.position.y), 1000, 50),
                              rl.Vector2(1000 / 2, 50 / 2), 0.0, rl.GREEN)

        if rl.is_key_down(rl.KeyboardKey.KEY_Q):
            sim.character_move_legs_q()
        elif rl.is_key_down(rl.KeyboardKey.KEY_W):
            sim.character_move_legs_w()

        if rl.is_key_down(rl.KeyboardKey.KEY_O):
            sim.character_move_knees_o()
        elif rl.is_key_down(rl.KeyboardKey.KEY_P):
            sim.character_move_knees_p()

        rl.end_mode_2d()

        # inputs = np.asarray(character_data_list(character))
        #
        # output = neural_network.feedforward(inputs)

        # def on_collision(arbiter, space, data):
        #     # Get the shapes that collided
        #     shape_1, shape_2 = arbiter.shapes
        #     list_of_shapes = [character.head.shape,
        #                       character.torso.shape,
        #                       character.right_biceps.limb.shape,
        #                       character.right_forearm.shape,
        #                       character.left_biceps.limb.shape,
        #                       character.left_forearm.shape]
        #
        #     # Check if the colliding shapes belong to the head and floor
        #     if (shape_1 in list_of_shapes and shape_2 == ground_shape) or (
        #             shape_1 == ground_shape and shape_2 in list_of_shapes):
        #         print("UPPER-BODY TOUCHED THE FLOOR")

        rl.draw_text("Distance: " + str(round(sim.draw_position().x, 0) / 1000.0) + "m", 20, 0, 50,
                     rl.Color(153, 204, 255, 255))
        rl.draw_text("Time: " + str(round(rl.get_time(), 2)), 20, 50, 50, rl.Color(153, 204, 255, 255))

        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
