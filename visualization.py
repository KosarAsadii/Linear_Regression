import matplotlib.pyplot as plt
import torch


# Visualization
def visualization(in_train, out_train, in_test, out_test, xr, train_prediction, results):
    w0_val = torch.linspace(-10, 10, 100)
    w1_val = torch.linspace(-10, 10, 100)
    w0, w1 = torch.meshgrid(w0_val, w1_val, indexing='ij')

    loss = torch.zeros_like(w0)

    for i in range(len(w0_val)):
        for j in range(len(w1_val)):
            w_0 = w0_val[i]
            w_1 = w1_val[j]
            loss[i, j] = torch.mean((w_0 + w_1 * in_train - out_train)** 2)
            

    fig = plt.figure()

    # Distribution of training and test data and model predictions
    ax1 = fig.add_subplot(221)
    ax1.scatter(in_train, out_train, label='Train')
    ax1.scatter(in_test, out_test, label='Test')
    ax1.plot(xr, train_prediction, label='Prediction', color='red')
    ax1.set_title('Distribution of training and test data and model predictions')
    ax1.legend()

    # Metrics Results
    ax4 = fig.add_subplot(222)
    ax4.axis('off')
    table = ax4.table(cellText=results.values, colLabels=results.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    colors = ['#f5f5f5', '#bdc574']
    for i, key in enumerate(table.get_celld().keys()):
        cell = table.get_celld()[key]
        cell.set_facecolor(colors[i % 2] if key[0] > 0 else '#93ab8c')
        cell.set_text_props(color='white' if key[0] == 0 else 'black')
        cell.set_edgecolor('#707b5b')

    plt.title("Metrics Results")
    plt.tight_layout()

    # 3D landscape of the loss function
    ax2 = fig.add_subplot(223, projection='3d')
    ax2.plot_surface(w0, w1, loss, cmap='viridis')
    ax2.set_title('3D landscape of the loss function')

    # 2D contour of the loss function
    ax3 = fig.add_subplot(224)
    cs = ax3.contour(w0, w1, loss, levels=torch.logspace(-2, 3, 20))
    ax3.clabel(cs, inline=1, fontsize=5)
    ax3.set_title('2D contour of the loss function')

    plt.show()
