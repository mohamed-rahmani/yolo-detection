from ultralytics import YOLO

model = YOLO("object-detection/best.pt")

results = model.track(source="images/photo_afp_boris_horvat_armes_saisie.webp", show=True, conf=0.4, save=True, project="runs")