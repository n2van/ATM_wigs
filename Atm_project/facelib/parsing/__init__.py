import torch

from facelib.utils import load_file_from_url
from .bisenet import BiSeNet
from .parsenet import ParseNet


def init_parsing_model(model_name='bisenet', half=False, device='cuda'):
    if model_name == 'bisenet':
        model = BiSeNet(num_class=19)
        model_url = 'https://cdn-lfs-us-1.hf.co/repos/a3/7f/a37ffc80afcedf1d6b3970f7c59503d4bf7ca6e4df17b8c45c942021e91cab5b/468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27parsing_bisenet.pth%3B+filename%3D%22parsing_bisenet.pth%22%3B&Expires=1746597023&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NjU5NzAyM319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2EzLzdmL2EzN2ZmYzgwYWZjZWRmMWQ2YjM5NzBmN2M1OTUwM2Q0YmY3Y2E2ZTRkZjE3YjhjNDVjOTQyMDIxZTkxY2FiNWIvNDY4ZTEzY2ExM2E5YjQzY2MwODgxYTlmOTkwODNhNDMwZTljMGEzOGFiZDkzNTQzMWQxYzI4ZWU5NGIyNjU2Nz9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=H0uMv2F9rkU00LUic6OhaCPDZRnHeG-xKVZ1lChshEtk9io3T2BcVRIu6PObBqLb0jD0JEy48d7FfarLOTtWixxjzHAcfYDpm0vpd9JY8gf0L0aGXqbqAJPXpXKKrr3PudAzmOIllZhjTJtcuU7eE7ELxs4A%7EhMaV9za20Zc1y6hQ5MjLas5oYGs8OyppxDx2%7E-Lp66AlLYm5aBnnMVfIF7AGkEImQFTWqppJo4cG2cHTJ%7E6ECxkMQKC2cfGuqIUw0KXE9Af9PB-He1UQe%7EhS68UtvzuZkvpTW-pfuk3QqtitUMOuemiNgx%7EtQAZSFoMi-M0RPbSwBBb%7Eqarv3G6Ig__&Key-Pair-Id=K24J24Z295AEI9'
    elif model_name == 'parsenet':
        model = ParseNet(in_size=512, out_size=512, parsing_ch=19)
        model_url = 'https://cdn-lfs-us-1.hf.co/repos/a3/7f/a37ffc80afcedf1d6b3970f7c59503d4bf7ca6e4df17b8c45c942021e91cab5b/3d558d8d0e42c20224f13cf5a29c79eba2d59913419f945545d8cf7b72920de2?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27parsing_parsenet.pth%3B+filename%3D%22parsing_parsenet.pth%22%3B&Expires=1746597043&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTc0NjU5NzA0M319LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmhmLmNvL3JlcG9zL2EzLzdmL2EzN2ZmYzgwYWZjZWRmMWQ2YjM5NzBmN2M1OTUwM2Q0YmY3Y2E2ZTRkZjE3YjhjNDVjOTQyMDIxZTkxY2FiNWIvM2Q1NThkOGQwZTQyYzIwMjI0ZjEzY2Y1YTI5Yzc5ZWJhMmQ1OTkxMzQxOWY5NDU1NDVkOGNmN2I3MjkyMGRlMj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=jJ4l9GsP26EJKrs2r8aaIxcZrXp5KWBMDeqf70Or8KTSvtZTYSO-zqKPRS3H0NhOmO50IBH2JHTQefofBR8ACgKAL1wA8KnaI3YtocYVpFGxZ3oEE9111ScuT3XdHAfU6qJJS8tTp6BrKTBfxdc7k-tK9ffyqI4w0rtvtTDNIwi1IcDLRjQLHkqCrKn6ra5d5nm6oYP-94Zh6BaRBJ4QuApPF%7EAU2DSmsyGramTLv7TxH%7EZZUYdNQ0lyojHHQvuludGJvoV21Ww7d%7EOsIY4SqVxEPrihdrGteD1EB0je6DZQIx6AgRfpYBKWCU%7E0uYOWtp8EYobto6rSyvFueNXbdg__&Key-Pair-Id=K24J24Z295AEI9'
    else:
        raise NotImplementedError(f'{model_name} is not implemented.')

    model_path = load_file_from_url(url=model_url, model_dir='weights/facelib', progress=True, file_name=None)
    load_net = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(load_net, strict=True)
    model.eval()
    model = model.to(device)
    return model
