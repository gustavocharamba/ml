import shutil
import numpy as np
import cv2
import os


def getFrames(video_path="video.mp4", frame_skip=120, patch_size=512, patches_per_frame=6):
    # Remover e recriar as pastas
    if os.path.exists("frames"):
        shutil.rmtree("frames")
    os.makedirs("frames")

    if os.path.exists("512"):
        shutil.rmtree("512")
    os.makedirs("512")

    capture = cv2.VideoCapture(video_path)
    aux = 0
    frame_count = 0

    while True:
        flag, frame = capture.read()
        if not flag:
            break

        if frame_count % frame_skip == 0:
            # Convertendo para escala de cinza
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Adicionando ruído normal
            noise = np.random.normal(60, 25, gray.shape)
            gray = np.clip(gray + noise, 150, 255).astype(np.uint8)

            # Salvando frame completo na pasta "frames"
            frame_name = os.path.join("frames", f'frame_{aux:04d}.png')
            cv2.imwrite(frame_name, gray)

            # Gerando patches aleatórios de 256x256
            h, w = gray.shape
            for i in range(patches_per_frame):
                # Escolhendo uma posição aleatória dentro do frame
                x = np.random.randint(0, max(1, w - patch_size))
                y = np.random.randint(0, max(1, h - patch_size))

                # Recortando a região 256x256
                patch = gray[y:y + patch_size, x:x + patch_size]

                # Salvando o patch na pasta "256"
                patch_name = os.path.join("512", f'frame_{aux:04d}_resized_{i}.png')
                cv2.imwrite(patch_name, patch)

            aux += 1

        frame_count += 1

    capture.release()
    print("Salvo.")


getFrames()
