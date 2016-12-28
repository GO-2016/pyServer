#!/usr/bin/python3
from __future__ import print_function
from message.message_pb2 import RequestV1, RequestV2, ResponseV1, ResponseV2
import numpy as np
import yaml
from keras.models import model_from_yaml, model_from_json
import os
import socket
from struct import pack

def load_model_and_weights(model_name = 'model.17.json', weights_name = 'weights.17.hd5'):
    with open(os.path.join('model', model_name), 'r') as f:
        #yml = yaml.load(f)
        #model = model_from_yaml(yaml.dump(yml))
        model = model_from_json(f.read())
        model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        model.load_weights(os.path.join('model', weights_name))
        return model

def reqv1_proto_to_np(proto):
    board_state = np.array(proto.board_state._values, dtype='<i4').reshape((19,19))
    our_lib1 = np.array(proto.our_group_lib1._values, dtype='<i4').reshape((19, 19))
    our_lib2 = np.array(proto.our_group_lib2._values, dtype='<i4').reshape((19, 19))
    our_lib3_plus = np.array(proto.our_group_lib3_plus._values, dtype='<i4').reshape((19, 19))
    oppo_lib1 = np.array(proto.oppo_group_lib1._values, dtype='<i4').reshape((19, 19))
    oppo_lib2 = np.array(proto.oppo_group_lib2._values, dtype='<i4').reshape((19, 19))
    oppo_lib3_plus = np.array(proto.oppo_group_lib3_plus._values, dtype='<i4').reshape((19, 19))
    return np.array([board_state, our_lib1, our_lib2, our_lib3_plus, oppo_lib1, oppo_lib2, oppo_lib3_plus])

def respv1_np_to_proto(arr):
    respV1 = ResponseV1()
    respV1.board_size = 361
    respV1.possibility.extend(arr.flatten().tolist())
    return respV1

def reqv2_proto_to_np(proto):
    board_state = np.array(proto.board_state._values, dtype='<i4').reshape((19,19))
    our_lib1 = np.array(proto.our_group_lib1._values, dtype='<i4').reshape((19, 19))
    our_lib2 = np.array(proto.our_group_lib2._values, dtype='<i4').reshape((19, 19))
    our_lib3 = np.array(proto.our_group_lib3._values, dtype='<i4').reshape((19, 19))
    our_lib4_plus = np.array(proto.our_group_lib4_plus._values, dtype='<i4').reshape((19, 19))
    oppo_lib1 = np.array(proto.oppo_group_lib1._values, dtype='<i4').reshape((19, 19))
    oppo_lib2 = np.array(proto.oppo_group_lib2._values, dtype='<i4').reshape((19, 19))
    oppo_lib3 = np.array(proto.oppo_group_lib3._values, dtype='<i4').reshape((19, 19))
    oppo_lib4_plus = np.array(proto.oppo_group_lib4_plus._values, dtype='<i4').reshape((19, 19))
    all_ones = np.array(proto.all_ones._values, dtype='<i4').reshape((19, 19))
    our_true_eye = np.array(proto.our_true_eye._values, dtype='<i4').reshape((19, 19))
    our_fake_eye = np.array(proto.our_fake_eye._values, dtype='<i4').reshape((19, 19))
    our_semi_eye = np.array(proto.our_semi_eye._values, dtype='<i4').reshape((19, 19))
    oppo_true_eye = np.array(proto.oppo_true_eye._values, dtype='<i4').reshape((19, 19))
    oppo_fake_eye = np.array(proto.oppo_fake_eye._values, dtype='<i4').reshape((19, 19))
    oppo_semi_eye = np.array(proto.oppo_semi_eye._values, dtype='<i4').reshape((19, 19))
    all_zeros = np.array(proto.all_zeros._values, dtype='<i4').reshape((19, 19))
    return np.array([board_state, our_lib1, our_lib2, our_lib3, our_lib4_plus,\
    				 oppo_lib1, oppo_lib2, oppo_lib3, oppo_lib4_plus,\
    				 all_ones, our_true_eye, our_fake_eye, our_semi_eye,\
    				 oppo_true_eye, oppo_fake_eye, oppo_semi_eye, all_zeros])

def respv2_np_to_proto(arr):
    respV2 = ResponseV2()
    respV2.board_size = 361
    respV2.possibility.extend(arr.flatten().tolist())
    return respV2

def handle_sock(sock, addr, model):
    size_bytes = sock.recv(8)
    size = int.from_bytes(size_bytes, byteorder='little')
    print('message size is ', size)
    message_bytes = sock.recv(size)

    #reqV1 = RequestV1()
    #reqV1.ParseFromString(message_bytes)
    reqV2 = RequestV2()
    reqV2.ParseFromString(message_bytes)

    if reqV2.board_size != 361:
        print('Invalid boardsize')
        sock.close()
        return

    np_arr_req = None
    try:
        np_arr_req = reqv2_proto_to_np(reqV2)
        #if np_arr_req.shape[0] == 7:
        #    np_arr_req = np.swapaxes(np_arr_req, 0, 2)
        #    np_arr_req = np.swapaxes(np_arr_req, 0, 1)
    except Exception as e:
        print('Invalid format: ', e)
        sock.close()
        return

    pred = model.predict(np.array([np_arr_req]))[0]
    print(pred)
    respV2 = respv2_np_to_proto(pred)

    str = respV2.SerializeToString()
    print('Sending back...')
    size_bytes = pack('<q', len(str))

    print('sending ', size_bytes)
    sock.send(size_bytes)
    sock.send(str)
    sock.close()

def run_server(model, port):
    print('Listening on port ', port)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(('', port))
    s.listen(64)

    while True:
        sock, addr = s.accept()
        print('Accepting new connection at ', addr)
        try:
            handle_sock(sock, addr, model)
        except Exception as e:
            print('Exception thrown')
            print(e)

if __name__ == '__main__':
    model = load_model_and_weights()
    run_server(model, 7391)
