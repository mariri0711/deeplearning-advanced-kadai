import os
from django.shortcuts import render
from .forms import ImageUploadForm
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from io import BytesIO

def predict(request):
    if request.method == 'GET':
        form = ImageUploadForm()
        return render(request, 'home.html', {'form': form})
    if request.method == 'POST':
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # 画像を取得
            img_file = form.cleaned_data['image']
            img_bytes = BytesIO(img_file.read())

            # VGG16が期待する224x224のサイズにリサイズ
            img = load_img(img_bytes, target_size=(224, 224))
            img_array = img_to_array(img)

            # 形状を (1, 224, 224, 3) に変換し、正規化
            img_array = img_array.reshape((1, 224, 224, 3))
            img_array = preprocess_input(img_array)

            # モデルをロード
            model_path = os.path.join(settings.BASE_DIR, 'prediction', 'models', 'vgg16.h5')
            model = load_model(model_path)

            # 予測の実行
            predictions = model.predict(img_array)

            # 上位5つのカテゴリを取得
            top_predictions = decode_predictions(predictions, top=5)[0]

            img_data = request.POST.get('img_data')
        else:
            form = ImageUploadForm()
    return render(request, 'home.html', {
        'form': form,
        'top_predictions': top_predictions,  # 上位5カテゴリをテンプレートに渡す
        'img_data': img_data
    })