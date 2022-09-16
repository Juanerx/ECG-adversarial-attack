from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
from Net import *


def perturb_data(xs, data):
    """

    :param xs: numpy array of perturbation(s), could be one perturbation eg.[16,0.1] or multiple perturbation eg.[[16,0.1],[...],...]
    :param data: a numpy array
    :return: perturbed data, the dimention of the array is 2D, either [[...]] or [[...],[...],...], numpy
    """
    # If this function is passed just one perturbation vector,
    # pack it in a list to keep the computation the same
    if xs.ndim < 2:
        xs = np.array([xs])

    # Copy the image n == len(xs) times so that we can
    # create n new perturbed data
    tile = [len(xs)] + [1] * (xs.ndim + 1)
    data_copies = np.tile(data, tile)

    # Make sure to floor the members of xs as int types
    # xs = xs.astype(float)

    for x, data in zip(xs, data_copies):
        # Split x into an array of 5-tuples (perturbation pixels)
        # i.e., [[x,y,r,g,b], ...]
        pixels = np.split(x, len(x) // 2)
        for pixel in pixels:
            # At each pixel's x,y position, assign its rgb value
            x_pos, value = pixel
            data[0,0,int(x_pos)] = value

    return data

def predict_classes(xs, data, target_class, model, minimize=True):
    """

    :param xs: numpy array of perturbation(s), could be one perturbation eg.[16,16,255,255,0] or multiple perturbation eg.[[16,16,255,255,0],[...],...]
    :param img: a numpy array, with pixel value range from -1 to 1
    :param target_class: the target class of the original image
    :param model: the model used to train the dataset
    :param minimize:
    :return: a list of the predicted confidence of the target class
    """
    # Perturb the image with the given pixel(s) x and get the prediction of the model
    data_perturbed = perturb_data(xs, data)
    input = torch.tensor(data_perturbed,dtype=torch.float32).to(device)
    predictions = F.softmax(model(input), dim=1).data.cpu().numpy()[:,target_class]
    # This function should always be minimized, so return its complement if needed
    return predictions if minimize else 1 - predictions

def predict(data, model):
    """
    Predict the confidence of the image that have gone through the model
    img: numpy -1~1

    """
    if data.ndim < 3:
        data = np.array([data])

    data = torch.tensor(data,dtype=torch.float32)
    data = data.to(device)
    confidence = F.softmax(model(data), dim=1).data.cpu().numpy()
    return confidence


def attack_success(x, data, target_class, model, targeted_attack=False, verbose=False):
    """

    :param x: numpy array of perturbation(s), could be one perturbation eg.[16,16,255,255,0] or multiple perturbation eg.[[16,16,255,255,0],[...],...]
    :param img: a numpy array, with pixel value range from -1 to 1
    :param target_class:
    :param model: the model used to train the dataset
    :param targeted_attack: if the attack is targeted
    :param verbose:
    :return: True or None due to if the attack is success or not
    """
    # Perturb the image with the given pixel(s) and get the prediction of the model
    attack_data = perturb_data(x, data)

    confidence = predict(attack_data,model)[0]
    predicted_class = np.argmax(confidence)
    # print('attacked predicted class: ',classes[predicted_class])

    # If the prediction is what we want (misclassification or
    # targeted classification), return True
    if verbose:
        print('Confidence:', confidence[target_class])
    if ((targeted_attack and predicted_class == target_class) or
            (not targeted_attack and predicted_class != target_class)):
        return True
    # NOTE: return None otherwise (not False), due to how Scipy handles its callback function


def attack(data, label, model, target=None, pixel_count=1,
           maxiter=75, popsize=400, verbose=False):
    # Change the target class based on whether this is a targeted attack or not
    targeted_attack = target is not None
    target_class = target if targeted_attack else label

    # Define bounds for a flat vector of x,y,r,g,b values
    # For more pixels, repeat this layout
    bounds = [(0, 250), (-1, 1)] * pixel_count

    # Population multiplier, in terms of the size of the perturbation vector x
    popmul = max(1, popsize // len(bounds))

    # Format the predict/callback functions for the differential evolution algorithm
    def predict_fn(xs):
        return predict_classes(xs, data, target_class,
                               model, target is None)

    def callback_fn(x, convergence):
        return attack_success(x, data, target_class,
                              model, targeted_attack,verbose)

    # Call Scipy's Implementation of Differential Evolution
    attack_result = differential_evolution(
        predict_fn, bounds, maxiter=maxiter, popsize=popmul,
        recombination=1, atol=-1, callback=callback_fn, polish=False)

    # Calculate some useful statistics to return from this function
    attack_data = perturb_data(attack_result.x, data)[0]
    prior_probs = predict(data,model)
    predicted_probs = predict(attack_data,model)
    predicted_class = np.argmax(predicted_probs)
    actual_class = label
    success = predicted_class != actual_class
    cdiff = prior_probs[0,actual_class] - predicted_probs[0,actual_class]

    # Show the best attempt at a solution (successful or not)
    # helper.plot_image(attack_image, actual_class, class_names, predicted_class)
    # plt.imshow(np.transpose(clipping(attack_image),(1,2,0)))
    # plt.show()
    if success == True:
        plot_paired(data,attack_data)

    return [pixel_count, actual_class, predicted_class, success, cdiff, prior_probs,
            predicted_probs, attack_result.x]


def plot_ecg(data):
    x_val = np.linspace(0,1000,250)
    y_val = data[0]
    plt.plot(x_val, y_val)
    # plt.show()

def plot_paired(data,attack_data):
    x_val = np.linspace(0,1000,250)
    fig, axes = plt.subplots(1, 2)
    axes[0].plot(x_val, data[0], label='Original_data')
    axes[0].legend()
    axes[0].set_xlabel('time')
    axes[0].set_ylabel('voltage')
    axes[1].plot(x_val, attack_data[0], label='Attack_data')
    axes[1].set_xlabel('time')
    axes[1].set_ylabel('voltage')
    axes[1].legend()
    plt.show()


if __name__ == '__main__':

    # Initialize model
    model = Net().to(device)

    # Load the pretraiined model
    pretrained_model = 'ecg_net.pth'
    model.load_state_dict(torch.load(pretrained_model,map_location='cpu'))
    model.eval()

    data = dataSet[0]
    label = labels[0]
    print(data)
    print('class:',classes[label])
    print('predict class:', predict_classes(np.array([0,1.0]),data,label,model))
    print(predict(data,model))
    print(attack_success(np.array([0,2]),data,label,model))

    success = 0
    for i in range(20):
        data = dataSet[i]
        label = labels[i]
        result = attack(data,label,model)[3]
        if result == True:
            print('Success')
            success += 1


        else:
            print('Fail')

    print('Success Rate:',success/10)

    # plot_ecg(data)