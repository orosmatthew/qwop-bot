import pyray as rl
import pymunk as pm
import json
import os

from character_simulation import CharacterSimulation


dir_count = 0
while True:
    if not os.path.exists(os.path.join('out', str(dir_count))):
        break
    dir_count += 1


# def output_data(gen_count: int, gen_list: list[CharacterSimulation]):
#     if not os.path.exists('out'):
#         os.mkdir('out')
#     if not os.path.exists(os.path.join('out', str(dir_count))):
#         os.mkdir(os.path.join('out', str(dir_count)))
#     data: list[dict] = []
#     for sim in gen_list:
#         data.append(sim.output_data())
#     with open(os.path.join('out', str(dir_count), str(gen_count) + ".json"), "w") as file:
#         json.dump(data, file)


def main():
    rl.set_target_fps(60)
    camera = rl.Camera2D(rl.Vector2(1280 / 2, 720 / 2), rl.Vector2(0, 0), 0.0, 1.0)
    rl.set_config_flags(rl.ConfigFlags.FLAG_MSAA_4X_HINT)
    rl.init_window(1280, 720, "QWOP-BOT")

    ground_position = 50, 150
    ground_poly = [
        (-50000, -25),
        (-50000, 25),
        (50000, 25),
        (50000, -25),
    ]

    ground_body: pm.Body = pm.Body(body_type=pm.Body.STATIC)
    ground_body.position = ground_position
    ground_shape = pm.Poly(ground_body, ground_poly)
    ground_shape.friction = 0.8
    ground_shape.collision_type = pm.Body.STATIC
    ground_shape.collision_type = 2

    # create 100 random characters for the 1st generation
    sim_list: list[CharacterSimulation] = [CharacterSimulation(ground_position, ground_poly) for _ in range(1)]

    sim_time: float = 0.0
    app_time: float = 0.0
    sub_sim_time: float = 0.0
    time_step = 1.0 / 60.0

    # Define the time intervals for updating the neural networks.
    # We break down 100 into 10s and 10s would take 3 sec. Set that way for debugging
    generation_duration = 30  # seconds
    subgen_duration = 3  # seconds

    # indexes to read 10 characters at a time
    # start_subgen: int = 0
    # end_subgen: int = 10
    #
    # sub_start_time = app_time
    # gen_count = 1
    # subgen_count = 1
    # subgen_num = 2
    #
    #
    # last_max = 0
    # last_max_time = 0
    #time given for sims to move past max distance after subgen_duration time
    subgen_duration_bonus = 3
    sim = CharacterSimulation(ground_position, ground_poly).draw_character()

    while not rl.window_should_close():
        # if rl.is_key_pressed(rl.KeyboardKey.KEY_S):
        #     data = []
        #     for sim in sim_list:
        #         data.append(sim.output_data())
        #     with open("network.json", "w") as file:
        #         json.dump(data, file)
        #
        # if rl.is_key_pressed(rl.KeyboardKey.KEY_L):
        #     with open("network.json", "r") as file:
        #         data = json.load(file)
        #     sim_list.clear()
        #     sim_list = [CharacterSimulation(ground_position, ground_poly) for _ in range(len(data))]
        #     sim_time = 0.0
        #     for i, sim in enumerate(sim_list):
        #         sim.load_data(data[i])
        #
        # if rl.is_key_pressed(rl.KeyboardKey.KEY_R):
        #     sim_list.clear()
        #     sim_list = [CharacterSimulation(ground_position, ground_poly) for _ in range(10)]
        #     sim_time = 0.0
        #
        for sim in sim_list:  # [start_subgen:end_subgen] so only work with 10 characters at a time
            sim.step(time_step)
            sim_time += time_step
            app_time += time_step
            sub_sim_time += time_step

        rl.begin_drawing()
        rl.begin_mode_2d(camera)

        rl.clear_background(rl.BLACK)

        # for sim in sim_list[
        #            start_subgen:end_subgen]:  # [start_subgen:end_subgen] so only work with 10 characters at a time
        #     sim.draw_character()
        CharacterSimulation(ground_position, ground_poly).draw_character()

        rl.draw_rectangle_pro(rl.Rectangle(round(ground_body.position.x), round(-ground_body.position.y), 50000, 50),
                              rl.Vector2(50000 / 2, 50 / 2), 0.0, rl.GREEN)

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

        max_x = -float('inf')
        # for sim in sim_list[
        #            start_subgen:end_subgen]:  # [start_subgen:end_subgen] so only work with 10 characters at a time
        #     sim.handler.separate = sim.collision_detection
        #     if sim.character_position().x > max_x:
        #         max_x = sim.character_position().x
        #         camera.target = sim.character_position()
        #     if max_x > last_max:
        #         last_max = max_x
        #         last_max_time = app_time



        # rl.draw_text("Max Distance: " + str(round(max_x, 0) / 1000.0) + "m", 20, 0, 50,
        #              rl.Color(153, 204, 255, 255))
        # rl.draw_text("Sim Time: " + str(round(sim_time, 2)), 20, 50, 50, rl.Color(153, 204, 255, 255))
        #
        # rl.draw_text("Gen: " + str(gen_count), 20, 100, 50, rl.Color(153, 204, 255, 255))
        #
        # rl.draw_text("SubGen: " + str(subgen_count), 20, 150, 50, rl.Color(153, 204, 255, 255))
        #
        # rl.draw_text("SubGen Time: " + str(round(sub_sim_time, 2)), 20, 200, 50, rl.Color(153, 204, 255, 255))

        rl.end_drawing()
    rl.close_window()


if __name__ == "__main__":
    main()
