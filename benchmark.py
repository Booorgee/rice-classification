# bench.py — save and run: python bench.py
import time
import torch
import timm

BATCH = 32
IMG = 224
N_ITERS = 10   # after warm-up

def bench(name, device, channels_last=False):
    torch.manual_seed(0)
    model = timm.create_model('tf_efficientnetv2_s.in1k',
                              pretrained=True, num_classes=2)
    if channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = model.to(device)

    # freeze backbone (head-only phase)
    for p in model.parameters():
        p.requires_grad = False
    for p in model.get_classifier().parameters():
        p.requires_grad = True

    def make_batch():
        x = torch.randn(BATCH, 3, IMG, IMG, device=device)
        if channels_last:
            x = x.to(memory_format=torch.channels_last)
        return x

    x = make_batch()
    y = torch.randint(0, 2, (BATCH,), device=device)
    loss_fn = torch.nn.CrossEntropyLoss()

    # warm-up (3 iters, not timed)
    try:
        for _ in range(3):
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
    except Exception as e:
        print(f"{name:35s} | ERROR: {type(e).__name__}: {str(e)[:80]}")
        return

    t = time.time()
    for _ in range(N_ITERS):
        out = model(x); loss = loss_fn(out, y); loss.backward()
    head = (time.time() - t) / N_ITERS

    # unfreeze all
    for p in model.parameters():
        p.requires_grad = True
    optim = torch.optim.AdamW(model.parameters(), lr=3e-5)

    try:
        for _ in range(3):
            optim.zero_grad()
            out = model(x); loss = loss_fn(out, y); loss.backward(); optim.step()
    except Exception as e:
        print(f"{name:35s} | head {head:.2f}s | full ERROR: {str(e)[:60]}")
        return

    t = time.time()
    for _ in range(N_ITERS):
        optim.zero_grad()
        out = model(x); loss = loss_fn(out, y); loss.backward(); optim.step()
    full = (time.time() - t) / N_ITERS

    print(f"{name:35s} | head {head:6.2f} s/batch | full {full:6.2f} s/batch")

# run all four combinations
bench("CPU  (default)",                 "cpu", channels_last=False)
bench("CPU  (channels_last)",           "cpu", channels_last=True)
bench("MPS  (default)",                 "mps", channels_last=False)
bench("MPS  (channels_last)",           "mps", channels_last=True)