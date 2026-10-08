import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_cake():
    fig, ax = plt.subplots()

    # Draw the cake layers
    cake_bottom = patches.Rectangle((0.1, 0.1), 0.8, 0.2, edgecolor='brown', facecolor='#D2691E')
    cake_middle = patches.Rectangle((0.15, 0.3), 0.7, 0.2, edgecolor='brown', facecolor='#F4A460')
    cake_top = patches.Rectangle((0.2, 0.5), 0.6, 0.2, edgecolor='brown', facecolor='#F5DEB3')

    ax.add_patch(cake_bottom)
    ax.add_patch(cake_middle)
    ax.add_patch(cake_top)

    # Draw the candles
    for i in range(5):
        candle = patches.Rectangle((0.25 + i * 0.1, 0.7), 0.05, 0.2, edgecolor='red', facecolor='pink')
        ax.add_patch(candle)

        # Draw the candle flames
        flame = patches.Ellipse((0.275 + i * 0.1, 0.92), 0.05, 0.1, edgecolor='yellow', facecolor='yellow')
        ax.add_patch(flame)

    # Draw the cake plate
    cake_plate = patches.Ellipse((0.5, 0.05), 0.9, 0.1, edgecolor='gray', facecolor='#D3D3D3')
    ax.add_patch(cake_plate)

    # Add "Happy Birthday" text
    plt.text(0.5, 0.95, 'Happy Birthday!', horizontalalignment='center', fontsize=15, color='purple')

    # Set the limits and aspect of the plot
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()

draw_cake()
