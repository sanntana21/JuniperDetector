from codetr.model import build_codetr

# Número de clases = 5 (tu dataset)
model = build_codetr(num_classes=5)
checkpoint = torch.load("pretrained.pth")
model.load_state_dict(checkpoint, strict=False)  # strict=False permite ignorar capa de salida
model = model.cuda()
