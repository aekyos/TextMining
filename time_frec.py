from lib import *
import os

def get_time(msg):
    return int(msg.split(",")[1].split(".")[0])

# DIR = "/tmp/test/"
DIR = "/home/pc/Documents/textmining/dota/purged_es/"

files = os.listdir(DIR)

sum = 0
number_of_files = 0

for filename in files:
    file = open(DIR + filename, "r")
    content = file.readlines()
    file.close()

    first_time = max(get_time(content[0]), 0)
    last_time = get_time(content[-1])
    game_duration = ((last_time-first_time) // 60) + 1 # time in minutes
    number_msgs = len(content)

    # cantidad de mensajes por minuto en el juego
    game_msg_frec = number_msgs / game_duration
    sum += game_msg_frec

    number_of_files += 1

# promedio de frecuencia de mensajes entre todas las partidas
avg_frecuency = sum/number_of_files
print("Games frec:", avg_frecuency)

def frec_desp_palabra(palabra):
    mensajes_despues = 0 # cantidad de mensajes despues del palabra
    for filename in files:
        file = open(DIR + filename, "r")
        content = file.readlines()
        file.close()
        N = len(content)

        palabra_in_game = False
        ventanas = []

        cant_msg = 0
        for i in range(N):
            msg = content[i]
            if palabra in msg:
                palabra_in_game = True
                time = get_time(msg=msg)

                in_ventanas = False
                for v in ventanas:
                    if v[0] < time and time <= v[1]:
                        in_ventanas = True
                        v[1] = time+60

                if not in_ventanas:
                    ventanas.append([time, time+60])

        if len(ventanas) > 0:
            for i in range(N):
                msg = content[i]
                time = get_time(msg=msg)

                msg_in_ventanas = False
                for v in ventanas:
                    if v[0] <= time and time <= v[1]:
                        msg_in_ventanas = True
                        break

                if msg_in_ventanas:
                    cant_msg += 1

            mensajes_despues += cant_msg / len(ventanas)

    return mensajes_despues

for desescalador in desescaladores:
    print(desescalador, frec_desp_palabra(palabra=desescalador))

print("")

expandidos = ['indio', 'mierda', 'idiota', 'void', 'mid', 'mono', 'ursa', 'salir', 'perder', 'venir', 'pasa', 'ganarian', 'rata', 'dar', 'dota', 'can', 'ez', 'noob', 'divine']
for expandido in expandidos:
    print(expandido, frec_desp_palabra(palabra=expandido))

print("")

for insulto in insultos:
    print(insulto, frec_desp_palabra(palabra=insulto))

print("")

for toxic in toxics:
    print(toxic, frec_desp_palabra(palabra=toxic))
