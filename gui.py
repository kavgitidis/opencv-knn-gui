import io
import os
import queue
import threading
import tkinter as tk
from pathlib import Path

import PySimpleGUI as psg
import numpy as np
from PIL import Image, UnidentifiedImageError

from utils import ImageUtils
from utils import MediaDB

root = tk.Tk()
screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
root.destroy()
width, height = 300, 300  # Scale image
psg.theme('Material2')
file_types = (("PNG (*.png)", "*.png"),
              ("JPEG (*.jpg)", "*.jpg"),
              ("All files (*.*)", "*.*"))
query_image = [

]

file_options = [
    [
        psg.Text("Image File"),
        psg.Input(size=(80, 1), disabled=True, key='-FILE-'),
        psg.Button('Browse'),
        psg.Button("Find KNNs"),
        psg.Button("Draw"),
    ],
    [
        psg.HorizontalSeparator()
    ],
    [
        psg.Text("Mongo Cluster"),
        psg.Input(size=(25, 1), key="-CLUSTER-", default_text="mongodb://localhost:27017/"),
        psg.Text("Mongo Database"),
        psg.Input(size=(10, 1), key="-DATABASE-", default_text='adb'),
        psg.Text("Mongo Collection"),
        psg.Input(size=(10, 1), key="-COLLECTION-", default_text='adb'),
        psg.Button("Add to DB"),
        psg.Button("Clear DB"),
    ],
    [
        psg.HorizontalSeparator()
    ],
    [
        psg.Text("K value"),
        psg.Combo([i + 1 for i in range(20)], size=(2, 5), key='k', default_value=[5]),
        psg.Text("Method for KNN"),
        psg.Combo(['FLANN', "BF"], size=(8, 2), key='method', default_value=['BF']),
        psg.Text("Algorithm"),
        psg.Combo(['SIFT', 'SURF', 'ORB', 'HISTOGRAM'], size=(12, 4), key='algorithm', default_value=['SIFT']),
        psg.Text("RATIO"),
        psg.Combo([round(i, 2) for i in np.arange(.9, 0, -.1)], size=(3, 6), key='ratio', default_value=[0.7]),
        psg.Text("Distance for\nHistogram"),
        psg.Combo(['euclidean', 'minkowski', 'jaccard', 'cosine', 'cityblock', 'dice', 'chebyshev', 'kulsinski'],
                  size=(10, 8), key='distance', default_value=['minkowski']),
        psg.Text("Distance for\nBFMatcher"),
        psg.Combo(['NORM_INF', 'NORM_L1', 'NORM_L2', 'NORM_L2SQR', 'NORM_HAMMING', 'NORM_HAMMING2', 'NORM_TYPE_MASK',
                   'NORM_RELATIVE', 'NORM_MINMAX'],
                  size=(18, 9), key='bfdist', default_value=['NORM_L2']),

    ],
    [psg.Text(key="-STATUS-", visible=False, text_color='red')],
    [psg.Image(key="-IMAGE-")],
]

knn_results = [[psg.Image(key=f"-IMAGE{int(i)}-"),
                psg.Image(key=f"-IMAGE{int(i + 1)}-"),
                psg.Image(key=f"-IMAGE{int(i + 2)}-"),
                psg.Image(key=f"-IMAGE{int(i + 3)}-"),
                psg.Image(key=f"-IMAGE{int(i + 4)}-"), ]
               if i == 1 or i % 5 == 1 else [
    psg.Text(key=f"-TEXT{int(i - 2.5)}-", size=(40, 2), justification='center', visible=False),
    psg.Text(key=f"-TEXT{int(i - 1.5)}-", size=(38, 2), justification='center', visible=False),
    psg.Text(key=f"-TEXT{int(i - .5)}-", size=(38, 2), justification='center', visible=False),
    psg.Text(key=f"-TEXT{int(i + .5)}-", size=(38, 2), justification='center', visible=False),
    psg.Text(key=f"-TEXT{int(i + 1.5)}-", size=(40, 2), justification='center', visible=False),
] for i in np.arange(1, 21, 2.5)]


def wrapper(file, method, algorithm, k, db, dist, ratio, bfdist, gui_queue):
    """

    Args:
        bfdist:
        ratio:
        dist:
        file:
        method:
        algorithm:
        k:
        db:
        gui_queue:

    Returns:

    """
    img_utils = ImageUtils()
    average_precision, knns = img_utils.get_knn(file, method, algorithm, k, db, dist, ratio, bfdist)
    gui_queue.put((average_precision, knns))
    return


def main():
    gui_queue = queue.Queue()
    layout = [[
        psg.Column(file_options),
        psg.VSeperator(),
        psg.Column(knn_results),
    ],
    ]

    window = psg.Window("Image Similarity", layout, resizable=True)

    flag = False
    matches = []
    while True:
        event, values = window.Read(timeout=100)
        if event == psg.WIN_CLOSED:
            break
        elif event == "Browse":
            window['-STATUS-'].update(visible=False)
            path = psg.popup_get_file("", no_window=True, file_types=file_types)
            if path == '':
                continue
            window['-FILE-'].update(path)
            if not Path(path).is_file():
                window['-STATUS-'].update('Image file not found!', visible=True)
                continue
            try:
                image = Image.open(path)

            except UnidentifiedImageError:
                window['-STATUS-'].update("Cannot identify image file!", visible=True)
                continue
            image = image.resize((400, 400))
            bio = io.BytesIO()
            image.save(bio, format="PNG")
            window[f"-IMAGE-"].update(data=bio.getvalue())
        elif event == "Clear DB":
            db = MediaDB(values["-CLUSTER-"], values["-DATABASE-"], values["-COLLECTION-"])
            db.collection.delete_many({})
            window['-STATUS-'].update("Cleared Database.", visible=True)
        elif event == "Add to DB":
            folder_path = psg.popup_get_folder("", no_window=True)
            db = MediaDB(values["-CLUSTER-"], values["-DATABASE-"], values["-COLLECTION-"])
            files = db.put_images_to_db(folder_path)
            window['-STATUS-'].update(f"Inserted {files} images to Database.", visible=True)
        elif event == "Find KNNs":
            flag = True
            for i in range(20):
                window[f"-IMAGE{i + 1}-"].update(visible=False)
                window[f"-TEXT{i + 1}-"].update(visible=False)
            window['-STATUS-'].update(visible=False)
            db = MediaDB(values["-CLUSTER-"], values["-DATABASE-"], values["-COLLECTION-"])
            thread_id = threading.Thread(target=wrapper, args=(
                values["-FILE-"], values["method"], values["algorithm"], int(values["k"]),
                db, values['distance'], float(values['ratio']), values['bfdist'], gui_queue), daemon=True)
            thread_id.start()
        elif event == "Draw":
            for i, img in enumerate(matches):
                if os.path.exists(img['ImagePath']):
                    image_utils = ImageUtils()
                    if values['algorithm'] == 'HISTOGRAM':
                        image = image_utils.draw_matches(img2=img['ImagePath'],
                                                         algorithm=values['algorithm'])
                    else:
                        image = image_utils.draw_matches(values["-FILE-"], img['ImagePath'],
                                                         image_utils.decode_keypoints(
                                                             img[f"Keypoints"]),
                                                         values['algorithm'],
                                                         img['Matches'], float(values['ratio']))
                    img = Image.fromarray(image)
                    img = img.resize((width, height))
                    bio = io.BytesIO()
                    img.save(bio, format="PNG")
                    window[f"-IMAGE{i + 1}-"].update(data=bio.getvalue())
                    window.move(0, 0)
                    if values["k"] > 16:
                        window.maximize()
                    else:
                        window.Normal()
        try:
            (average_precision, knns) = gui_queue.get_nowait()  # see if something has been posted to Queue
        except queue.Empty:  # get_nowait() will get exception when Queue is empty
            knns = None  # nothing in queue so do nothing
        if knns is not None:
            flag = False
            matches = knns
            for i, img in enumerate(knns):
                if os.path.exists(img['ImagePath']):
                    image = Image.open(img['ImagePath'])
                    image = image.resize((width, height))
                    bio = io.BytesIO()
                    image.save(bio, format="PNG")
                    window[f"-IMAGE{i + 1}-"].update(data=bio.getvalue())
                    window[f"-IMAGE{i + 1}-"].update(visible=True)
                    window.Element(f"-IMAGE{i + 1}-").set_tooltip(f"{round(img['Precision'], 4)}%")
                    window[f"-TEXT{i + 1}-"].update(f"{img['ImageName']}\n{round(img['Precision'], 4)}%")
                    window[f"-TEXT{i + 1}-"].update(visible=True)
                    # window.Element(f"-TEXT{i + 1}-").set_visible(True)
                    window.move(0, 0)
                    window['-STATUS-'].update(f"Average precision: {round(average_precision, 4)}%", visible=True)
                    if values["k"] > 16:
                        window.maximize()
                    else:
                        window.Normal()
        if flag:
            psg.PopupAnimated(psg.DEFAULT_BASE64_LOADING_GIF, time_between_frames=100,
                              message=f'Querying the {values["k"]} most similar images.')
        else:
            psg.PopupAnimated(None)
    window.close()


if __name__ == "__main__":
    main()
