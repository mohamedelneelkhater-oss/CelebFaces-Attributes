import torch

from gan_faces import Generator, discover_models, generate_faces, load_generator


def test_generator_output_shape_and_range():
    faces = generate_faces(Generator(), 4, seed=1)
    assert faces.shape == (4, 3, 64, 64)
    assert faces.min() >= -1.0 and faces.max() <= 1.0


def test_same_seed_gives_same_faces():
    G = Generator()
    assert torch.equal(generate_faces(G, 2, seed=7), generate_faces(G, 2, seed=7))


def test_checkpoint_roundtrip_and_discovery(tmp_path):
    G = Generator(latent_dim=100, features=64)
    torch.save(
        dict(name="DCGAN", loss_type="bce", state_dict=G.state_dict(), latent_dim=100,
             gen_features=64, epoch=12, metric=1.5),
        tmp_path / "DCGAN_best.pth",
    )
    found = discover_models(tmp_path)
    assert list(found) == ["DCGAN"] and found["DCGAN"]["epoch"] == 12
    loaded, ckpt = load_generator(found["DCGAN"]["path"])
    assert ckpt["metric"] == 1.5
    assert torch.equal(generate_faces(G, 2, seed=3), generate_faces(loaded, 2, seed=3))
