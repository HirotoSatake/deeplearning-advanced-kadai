from django.shortcuts import render
from .forms import ImageUploadForm
from tensorflow.keras.preprocessing.image import load_img, img_to_array # type: ignore
from tensorflow.keras.applications.vgg16 import preprocess_input, VGG16, decode_predictions # type: ignore
from io import BytesIO

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})

    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            img_file = form.cleaned_data['image']

            # 画像データをメモリに読み込む
            img_file = BytesIO(img_file.read())
            img = load_img(img_file, target_size=(224, 224))
            img_array = img_to_array(img)
            img_array = img_array.reshape((1, 224, 224, 3))

            img_array = preprocess_input(img_array)

            # VGG16モデルを読み込む
            model = VGG16(weights='imagenet')

            # 予測を実行
            preds = model.predict(img_array)
            prediction = decode_predictions(preds, top=5)[0]  # 上位5つの予測を取得

            return render(request, 'home.html', {'form': form, 'prediction': prediction})

    else:
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
