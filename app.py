from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64
import numpy as np
from PIL import Image
from io import BytesIO
import queue
from threading import Thread, Event
import torch
import time
from utils import get_data
from algorithm import Algorithm
from models.vae.sdvae import SDVAE
import config.configTrain as cfg
from infer_test import init_simulator, get_web_img,get_img_data
import os
"""Model info"""
frame_rate = 1 / 20

model = Algorithm(cfg.model_name, cfg.device)
state_dict = torch.load(os.path.join("ckpt",cfg.model_path),map_location=cfg.device,weights_only=False)
model.load_state_dict(state_dict["network_state_dict"],strict=False)
model.eval().to(cfg.device)
vae = SDVAE()
vae_state_dict = torch.load(cfg.vae_model, map_location=cfg.device,weights_only=False)
vae.load_state_dict(vae_state_dict['network_state_dict'], strict=False)
vae.eval().to(cfg.device)



init_data = get_data()

# with torch.no_grad():
#     init_obs, init_zeta = init_simulator(model, init_data)


app = Flask(__name__)
socketio = SocketIO(app,
                    cors_allowed_origins="*",
                    logger=False,
                    engineio_logger=False,
                    transports=['websocket', 'polling'],
                    allow_upgrades=True)

"""User Info"""
user_zeta = {}
user_numpy_data = {}
user_queues = {}
user_cmd = {}
user_config = {}
user_threads = {}
online_player = {}

default_cmd_str = "NULL"  # 与前端保持一致


def numpy2imgstr(image):
    image = get_web_img(image)
    img = Image.fromarray(image)
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return img_str


def get_jave_7action(key):
    if key == "r":
        action = 2
    elif key == "rj":
        action = 6
    elif key == "l":
        action = 1
    elif key == "lj":
        action = 5
    elif key == "j":
        action = 4
    elif key == "s":
        action = 8
    elif key == "NULL" or key == "None":  # 兼容两种写法
        action = 0
    else:
        action = 0
    return [action]


def get_user_config(data):
    ret = {
        'random_init': False,
        'block': False,
        'denosing_step': 4,
    }
    random_init = data['random_init']
    block = data['block']
    denosing_step = data['denosing_step']
    if random_init == 'true':
        ret['random_init'] = True
    if block == 'true':
        ret['block'] = True
    if denosing_step == '8':
        ret['denosing_step'] = 8
    return ret


@socketio.on('key_press')
def handle_key_press(data):
    user_id = request.sid
    key = data['key']
    user_cmd[user_id] = key


@socketio.on('key_release')
def handle_key_release(data):
    user_id = request.sid
    key = data['key']
    if user_cmd[user_id] == 'lj':
        if key == 'l':
            user_cmd[user_id] = 'j'
        elif key == 'j':
            user_cmd[user_id] = 'l'
    elif user_cmd[user_id] == 'rj':
        if key == 'r':
            user_cmd[user_id] = 'j'
        elif key == 'j':
            user_cmd[user_id] = 'r'
    else:
        user_cmd[user_id] = default_cmd_str


@socketio.on('start_game')
def button_clicked(data):
    user_id = request.sid
    player_config = get_user_config(data)
    user_config[user_id] = player_config
    
    # 重置游戏状态：清空之前的命令和状态
    user_cmd[user_id] = default_cmd_str  # 重置按键命令
    
    # init simulator first frame
    with torch.no_grad():
        if player_config['random_init']:
            batch_data = get_data(if_random=True)
        else:
            # 创建副本避免修改全局 init_data
            batch_data = {"observations": init_data["observations"].clone()}
            if "cur_actions" in init_data:
                batch_data["cur_actions"] = init_data["cur_actions"].clone()
        
        # 转换数据格式：从 [1, 1, 3, 256, 256] 转为 [1, 3, 256, 256]（与get_img_data格式一致）
        if batch_data["observations"].dim() == 5:
            batch_data["observations"] = batch_data["observations"].squeeze(1)  # [1, 1, 3, 256, 256] -> [1, 3, 256, 256]
        
        zeta,obs = init_simulator(model, vae, batch_data)

    user_zeta[user_id] = zeta  # 重置zeta状态
    # decoded_obs 形状是 [1, 3, 256, 256]，obs[0] 得到 [3, 256, 256]
    user_queues[user_id].put((obs[0].cpu().numpy(), "None", "init"))

    # record player
    online_player[user_id] = user_id
    socketio.emit('update_person', {'num': len(online_player)})


# @socketio.on('start_game')
# def button_clicked(data):
#     user_id = request.sid
#     player_config = get_user_config(data)
#     user_config[user_id] = player_config
#
#     if player_config['random_init']:
#         random_data = get_data(if_random=True)
#         user_numpy_data[user_id] = random_data
#     else:
#         user_numpy_data[user_id] = init_data
#
#     online_player[user_id] = user_id
#     socketio.emit('update_person', {'num': len(online_player)})

def model_inference(user_id, stop_event):
    while not stop_event.is_set():
        if user_id in user_numpy_data.keys():
            start_time = time.time()
            with torch.no_grad():
                zeta,obs = init_simulator(model, vae, user_numpy_data[user_id])
            user_zeta[user_id] = zeta
            end_time = time.time()
            obs = obs[0].cpu().numpy()
            duration = f"{end_time - start_time:.2f} second"
            rest_time = frame_rate - (end_time - start_time)
            if rest_time > 0:
                time.sleep(rest_time)

            user_queues[user_id].put((obs, "None", duration))
            user_numpy_data.pop(user_id, None)
        elif user_id in user_cmd.keys() and user_id in user_zeta.keys():
            start_time = time.time()
            key = user_cmd[user_id]
            block = user_config[user_id]['block']
            sampling_timesteps = user_config[user_id]['denosing_step']
            if key == default_cmd_str and block:
                continue

            action = get_jave_7action(key)
            action = torch.tensor(action, device=cfg.device).long()
            zeta = user_zeta[user_id]
            with torch.no_grad():
                # Diffusion step
                step_start = time.time()
                zeta, obs = model.df_model.step(zeta, action.float(), sampling_timesteps)
                step_time = time.time() - step_start
                
                # VAE decode
                decode_start = time.time()
                obs = vae.decode(obs / 0.1355)
                decode_time = time.time() - decode_start
                
            user_zeta[user_id] = zeta
            # 转移到CPU并转换numpy（只在最后做一次）
            transfer_start = time.time()
            obs = obs[0].cpu().numpy()
            transfer_time = time.time() - transfer_start
            
            end_time = time.time()
            total_time = end_time - start_time
            # 显示详细时间分解（可选，用于调试）
            duration = f"{total_time:.3f}s (step:{step_time:.3f} decode:{decode_time:.3f} transfer:{transfer_time:.3f})"
            rest_time = frame_rate - (end_time - start_time)
            if rest_time > 0:
                time.sleep(rest_time)

            user_queues[user_id].put((obs, key, duration))
        else:
            time.sleep(0.01)


def send_results(user_id, stop_event):
    while not stop_event.is_set():
        if not user_queues[user_id].empty():
            obs, cmd_str_, dur = user_queues[user_id].get()
            socketio.emit('update_frame', {
                'image': numpy2imgstr(obs),
            }, room=user_id)
            socketio.emit('update_cmd', {'cmd': cmd_str_, 'dur': dur}, room=user_id)
        else:
            time.sleep(0.01)


@socketio.on('connect')
def handle_connect():
    user_id = request.sid
    user_cmd[user_id] = default_cmd_str
    user_queues[user_id] = queue.Queue()
    join_room(user_id)

    stop_event = Event()
    inference_thread = Thread(target=model_inference, args=(user_id, stop_event))
    inference_thread.daemon = True
    inference_thread.start()

    result_thread = Thread(target=send_results, args=(user_id, stop_event))
    result_thread.daemon = True
    result_thread.start()

    user_threads[user_id] = (inference_thread, result_thread, stop_event)
    socketio.emit('update_person', {'num': len(online_player)})


@socketio.on('disconnect')
def handle_disconnect():
    user_id = request.sid
    leave_room(user_id)
    user_cmd.pop(user_id, None)
    user_queues.pop(user_id, None)
    user_config.pop(user_id, None)
    online_player.pop(user_id, None)
    if user_id in user_threads:
        inference_thread, result_thread, stop_event = user_threads.pop(user_id)
        stop_event.set()
        inference_thread.join()
        result_thread.join()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/debug')
def debug():
    return render_template('debug_test.html')


@app.route('/test')
def test():
    return "Flask应用正常工作！"


@socketio.on('test')
def handle_test(data):
    emit('test_response', {'message': '测试成功', 'received': data})


if __name__ == '__main__':
    import os

    # 检查是否在Colab环境中运行
    try:
        import google.colab

        IN_COLAB = True
    except ImportError:
        IN_COLAB = False

    if IN_COLAB:
        # 方法1：尝试使用ngrok（需要token）
        ngrok_success = False
        try:
            from pyngrok import ngrok

            # 检查是否有ngrok token
            ngrok_token = os.environ.get('NGROK_AUTHTOKEN')
            if ngrok_token:
                ngrok.set_auth_token(ngrok_token)

            # 启动ngrok隧道
            public_url = ngrok.connect(1235)
            print(f"应用已启动！")
            print(f"本地地址: http://localhost:1235")
            print(f"公网地址: {public_url}")
            print(f"请在浏览器中打开: {public_url}")
            ngrok_success = True

        except Exception as e:
            print("ngrok启动失败，使用Colab内置端口转发")
            print("请使用: google.colab.kernel.proxyPort(1235) 获取公网URL")

        # 在Colab中运行
        socketio.run(app, debug=False, host='0.0.0.0', port=1235, allow_unsafe_werkzeug=True)

    else:
        # 本地运行
        socketio.run(app, debug=True, host='0.0.0.0', port=1235)
