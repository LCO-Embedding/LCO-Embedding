from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch

image_embeddings = torch.load('./embeddings/image_embeddings_pixmo_omni.pt') # [layer_num, example_num, dimensionality], e.g., 28 * 1000 * 3584
text_embeddings = torch.load('./embeddings/text_embeddings_pixmo_omni.pt')

average_lower_triangles = []
for layer in range(image_embeddings.shape[0]):
    layer_average = []
    for embeddings in [image_embeddings,text_embeddings]:
        cossim_matrix = cosine_similarity(embeddings[layer].float(),embeddings[layer].float())
        lower_triangle = np.tril(cossim_matrix, k=-1)
        elements_count = np.count_nonzero(lower_triangle)
        average_lower_triangle = np.sum(lower_triangle) / elements_count
        layer_average.append(average_lower_triangle)
    average_lower_triangles.append(layer_average)

# the `average_lower_triangles` will now be the shape of (layer_num, 2), where 2 is the value for image and the value of text.


# do the same for audio, video, e.g.,
# audio_embeddings = torch.load('./embeddings/image_embeddings_audiocaps_omni.pt') # [layer_num, example_num, dimensionality], e.g., 28 * 1000 * 3584
# text_embeddings = torch.load('./embeddings/text_embeddings_audiocaps_omni.pt')


# video_embeddings = torch.load('./embeddings/image_embeddings_msrvtt_omni.pt') # [layer_num, example_num, dimensionality], e.g., 28 * 1000 * 3584
# text_embeddings = torch.load('./embeddings/text_embeddings_msrvtt_omni.pt')
