import os
import torch
from torch import nn
from copy import deepcopy

from facelib.utils import load_file_from_url
from facelib.utils import download_pretrained_models
from facelib.detection.yolov5face.models.common import Conv

from .retinaface.retinaface import RetinaFace
from .yolov5face.face_detector import YoloDetector


def init_detection_model(model_name, half=False, device='cuda'):
    if 'retinaface' in model_name:
        model = init_retinaface_model(model_name, half, device)
    elif 'YOLOv5' in model_name:
        model = init_yolov5face_model(model_name, device)
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    return model


def init_retinaface_model(model_name, half=False, device='cuda'):
    if model_name == 'retinaface_resnet50':
        model = RetinaFace(network_name='resnet50', half=half)
        model_url = 'https://cdn-lfs-us-1.hf.co/repos/a3/7f/a37ffc80afcedf1d6b3970f7c59503d4bf7ca6e4df17b8c45c942021e91cab5b/6d1de9c2944f2ccddca5f5e010ea5ae64a39845a86311af6fdf30841b0a5a16d?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27detection_Resnet50_Final.pth%3B+filename%3D%22detection_Resnet50_Final.pth%22%3B&Expires=1746596689&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NjU5NjY4OX19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2EzLzdmL2EzN2ZmYzgwYWZjZWRmMWQ2YjM5NzBmN2M1OTUwM2Q0YmY3Y2E2ZTRkZjE3YjhjNDVjOTQyMDIxZTkxY2FiNWIvNmQxZGU5YzI5NDRmMmNjZGRjYTVmNWUwMTBlYTVhZTY0YTM5ODQ1YTg2MzExYWY2ZmRmMzA4NDFiMGE1YTE2ZD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=IZgaK72AL8XiG7nj3OadGvMuia55LBzb%7Es%7EGvoXZ63bk9YHSdYRSMISo-TjUKb%7Eo79J8WrH846DMKTTW8dhWkX8T%7EzW97NIYS-RV8GN3p-amBLFREXl7gMO7t9G%7E%7ETbqR4FxQggh6zVT1E%7EGfoEeS7x8eh23G4pQCQxUc24xypsG7UyQK6FaIPSal09MmF8wRrxjwn3YQZHz8GFeBJhAsj0PAwM%7Ev5t84A0RGSAM7tMj6EwONkEvgTEuSRJMJDKaIHoN9J2ZE5sFogmh1C563JX8MsL2jdJyuTbN11ov7yvoUcSnEulL-3aQxQHIcN-%7EZP1nrqOM29sSyddB9OKSUw__&Key-Pair-Id=K24J24Z295AEI9'
    elif model_name == 'retinaface_mobile0.25':
        model = RetinaFace(network_name='mobile0.25', half=half)
        model_url = 'https://cdn-lfs-us-1.hf.co/repos/a3/7f/a37ffc80afcedf1d6b3970f7c59503d4bf7ca6e4df17b8c45c942021e91cab5b/2979b33ffafda5d74b6948cd7a5b9a7a62f62b949cef24e95fd15d2883a65220?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27detection_mobilenet0.25_Final.pth%3B+filename%3D%22detection_mobilenet0.25_Final.pth%22%3B&Expires=1746596723&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NjU5NjcyM319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2EzLzdmL2EzN2ZmYzgwYWZjZWRmMWQ2YjM5NzBmN2M1OTUwM2Q0YmY3Y2E2ZTRkZjE3YjhjNDVjOTQyMDIxZTkxY2FiNWIvMjk3OWIzM2ZmYWZkYTVkNzRiNjk0OGNkN2E1YjlhN2E2MmY2MmI5NDljZWYyNGU5NWZkMTVkMjg4M2E2NTIyMD9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=O2NBQvX-h2Fovc4AHqfNrI7v2WAuJIVmObB9cLNjviMF42uL0vdoDBAi4XhOHosK5cX3gwK6ILssYWIfjKi1%7EWURqgOcASe4vrlyDrdoMtKKv%7EGZ8TmeV1XjhKaiapB8ZTOy9wochhvF11mHwo1HD%7Ea4H%7EKE6J2IvDmLuYwBpiDvsgbZ3jm8fPqEzLYqJrg9LLO9i3MgdNT87w6S8SyY%7ESocTUQ69ip0rE-TIY0us0JeLN2olY0kaElfBitWzzEHXwKiHkp4sUn81PxYuiUC9z3EeGo9u61qb0AW-OFe0W3--pIWAbWQFHeZe-e7UZNQhKDSKZicVSGT53ePlvrsdg__&Key-Pair-Id=K24J24Z295AEI9'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='weights/facelib', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    # remove unnecessary 'module.'
    for k, v in deepcopy(load_net).items():
        if k.startswith('module.'):
            load_net[k[7:]] = v
            load_net.pop(k)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)

    return model


def init_yolov5face_model(model_name, device='cuda'):
    if model_name == 'YOLOv5l':
        model = YoloDetector(config_name='facelib/detection/yolov5face/models/yolov5l.yaml', device=device)
        model_url = 'https://cdn-lfs-us-1.hf.co/repos/a3/7f/a37ffc80afcedf1d6b3970f7c59503d4bf7ca6e4df17b8c45c942021e91cab5b/1ba9d2125fda4d823df5152b9fc2903c59aa76c0d3771e02bcf13a56a282cf96?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27yolov5l-face.pth%3B+filename%3D%22yolov5l-face.pth%22%3B&Expires=1746596792&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NjU5Njc5Mn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2EzLzdmL2EzN2ZmYzgwYWZjZWRmMWQ2YjM5NzBmN2M1OTUwM2Q0YmY3Y2E2ZTRkZjE3YjhjNDVjOTQyMDIxZTkxY2FiNWIvMWJhOWQyMTI1ZmRhNGQ4MjNkZjUxNTJiOWZjMjkwM2M1OWFhNzZjMGQzNzcxZTAyYmNmMTNhNTZhMjgyY2Y5Nj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=la9P1BR5AMwP2fv67SNBozl%7EtyzY5R2lYV4LjlVv2BxcX4-3M3MYc1rFzZl3iyxTluVXC2NIyu1aG%7EO9X75kLeJzFkKiE-Y-Oxpl8BtpUP5zasVBEvIihJLQ0jw3KwgfdxOOFGxH%7ECka50JAVR0Lsm0FWvaN6kpLVL0mrr3-cKfV8J3yZgdDuAElbX0Sg%7EJ54KaGJLh8JYjtmCvk7frhH7iw7DHbLuKw-1pzfaBL%7E-Lt9mzZIXB%7Ej4PAVRUUSdMBIW3jnE5Ab3QoVeUQd76dGCHK8C0gC9wZlPbbSc8rNmnxANoSNxH4yP9HERBpDNeGXbaPUez9kDAENdEVoRheog__&Key-Pair-Id=K24J24Z295AEI9'
    elif model_name == 'YOLOv5n':
        model = YoloDetector(config_name='facelib/detection/yolov5face/models/yolov5n.yaml', device=device)
        model_url = 'https://cdn-lfs-us-1.hf.co/repos/a3/7f/a37ffc80afcedf1d6b3970f7c59503d4bf7ca6e4df17b8c45c942021e91cab5b/d2bbfbe9f36cf1ec345dc69658d7209e5448a676d946f1bf7818ac50d4489357?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27yolov5n-face.pth%3B+filename%3D%22yolov5n-face.pth%22%3B&Expires=1746596812&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NjU5NjgxMn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2EzLzdmL2EzN2ZmYzgwYWZjZWRmMWQ2YjM5NzBmN2M1OTUwM2Q0YmY3Y2E2ZTRkZjE3YjhjNDVjOTQyMDIxZTkxY2FiNWIvZDJiYmZiZTlmMzZjZjFlYzM0NWRjNjk2NThkNzIwOWU1NDQ4YTY3NmQ5NDZmMWJmNzgxOGFjNTBkNDQ4OTM1Nz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=dcKDDOg4niV44unkZcKF15SI9qzHwSLRo859DiAJ9Po1eaFXTKajccAjxvFrDc9%7EpsyjSM8gpSNfkNBAxwpdjajgPtlZbmuTTkXx%7Eoy1r5kFraFiOgMnVfIobj0Gn4vQybQpFx7ervy2TY27sb-ik1x12fCq719zoOKzxO-k5UOl5j5Na8lw0U%7EyguCIjr0ewxTm39p3bBj72udLZjqdtHMEjtcu39459-jEcP7diffYEiesdpeRQyBONbcQo2kd3u9qDsYaPOZ8SD%7Eb7mtKOxs14dPMMv4TVzAExwDFtNOcmofZwfnux%7ELe5v8B9tpjoinuTHVxLmAsXs2kectxpQ__&Key-Pair-Id=K24J24Z295AEI9'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')
    
    model_path = load_file_from_url(url=model_url, model_dir='weights/facelib', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.detector.load_state_dict(load_net, strict=True)
    model.detector.eval()
    model.detector = model.detector.to(device).float()

    for m in model.detector.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif isinstance(m, Conv):
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    return model


# Download from Google Drive
# def init_yolov5face_model(model_name, device='cuda'):
#     if model_name == 'YOLOv5l':
#         model = YoloDetector(config_name='facelib/detection/yolov5face/models/yolov5l.yaml', device=device)
#         f_id = {'yolov5l-face.pth': '131578zMA6B2x8VQHyHfa6GEPtulMCNzV'}
#     elif model_name == 'YOLOv5n':
#         model = YoloDetector(config_name='facelib/detection/yolov5face/models/yolov5n.yaml', device=device)
#         f_id = {'yolov5n-face.pth': '1fhcpFvWZqghpGXjYPIne2sw1Fy4yhw6o'}
#     else:
#         raise NotImplementedError(f'{model_name} is not implemented.')

#     model_path = os.path.join('weights/facelib', list(f_id.keys())[0])
#     if not os.path.exists(model_path):
#         download_pretrained_models(file_ids=f_id, save_path_root='weights/facelib')

#     load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
#     model.detector.load_state_dict(load_net, strict=True)
#     model.detector.eval()
#     model.detector = model.detector.to(device).float()

#     for m in model.detector.modules():
#         if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
#             m.inplace = True  # pytorch 1.7.0 compatibility
#         elif isinstance(m, Conv):
#             m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

#     return model