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
    def __init__(self, ground_position: tuple[float, float], ground_poly: list[tuple[float, float]]):
        self.space: pm.Space = pm.Space()
        self.space.gravity = (0, -900.0)

        self.character: Character = Character(self.space, leg_muscle_strength=1_000_000.0, arm_muscle_strength=50_000.0)

        ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
        ground_body.position = ground_position
        ground_shape = pm.Poly(ground_body, ground_poly)
        ground_shape.friction = 0.8
        ground_shape.collision_type = pm.Body.STATIC

        self.space.add(ground_body, ground_shape)

        self.neural_network: NeuralNetwork = NeuralNetwork()

        self.outputs = np.asarray(character_data_list(self.character))

        self.fitness = 0.0

    def step(self, time_step: float) -> None:
        self.space.step(time_step)
        inputs = np.asarray(character_data_list(self.character))
        self.outputs = self.neural_network.feedforward(inputs)
        if self.outputs[0] >= 0.5:
            self.character_move_legs_q()
        elif self.outputs[1] >= 0.5:
            self.character_move_legs_w()
        if self.outputs[2] >= 0.5:
            self.character_move_knees_o()
        elif self.outputs[3] >= 0.5:
            self.character_move_knees_p()

        fitness = self.character_position().x

    def character_position(self) -> rl.Vector2:
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

    ground_position = 300, 150
    ground_poly = [
        (-500, -25),
        (-500, 25),
        (500, 25),
        (500, -25),
    ]

    sim_list: list[CharacterSimulation] = [CharacterSimulation(ground_position, ground_poly) for _ in range(10)]

    ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
    ground_body.position = ground_position
    ground_shape = pm.Poly(ground_body, ground_poly)
    ground_shape.friction = 0.8
    ground_shape.collision_type = pm.Body.STATIC

    sim_time: float = 0.0
    time_step = 1.0 / 60.0

    while not rl.window_should_close():
        if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
            sim_list.clear()
            sim_list = [CharacterSimulation(ground_position, ground_poly) for _ in range(10)]
            sim_time = 0.0

        for sim in sim_list:
            sim.step(time_step)
        sim_time += time_step

        rl.begin_drawing()
        rl.begin_mode_2d(camera)

        rl.clear_background(rl.BLACK)

        for sim in sim_list:
            sim.draw_character()

        rl.draw_rectangle_pro(rl.Rectangle(round(ground_body.position.x), round(-ground_body.position.y), 1000, 50),
                              rl.Vector2(1000 / 2, 50 / 2), 0.0, rl.GREEN)

        # if rl.is_key_down(rl.KeyboardKey.KEY_Q):
        #     sim.character_move_legs_q()
        # elif rl.is_key_down(rl.KeyboardKey.KEY_W):
        #     sim.character_move_legs_w()
        #
        # if rl.is_key_down(rl.KeyboardKey.KEY_O):
        #     sim.character_move_knees_o()
        # elif rl.is_key_down(rl.KeyboardKey.KEY_P):
        #     sim.character_move_knees_p()

        rl.end_mode_2d()

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

        max_x = -float('inf')
        for sim in sim_list:
            if sim.character_position().x > max_x:
                max_x = sim.character_position().x
                camera.target = sim.character_position()

        rl.draw_text("Max Distance: " + str(round(max_x, 0) / 1000.0) + "m", 20, 0, 50,
                     rl.Color(153, 204, 255, 255))
        rl.draw_text("Sim Time: " + str(round(sim_time, 2)), 20, 50, 50, rl.Color(153, 204, 255, 255))

        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
