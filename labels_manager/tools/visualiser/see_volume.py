import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons


def see_array(in_array, extra_image=None, scale=None, num_fig=1, block=False,
              title='Image in matrix coordinates, C convention.'):

    fig = plt.figure(num_fig, figsize=(6, 7.5), dpi=100)
    ax = fig.add_subplot(111)
    ax.set_position([0.1, 0.29, 0.8, 0.7])

    fig.canvas.set_window_title(title)

    dims = in_array.shape  # (i,j,k,t,d)
    dims_mean = [int(d / 2) for d in dims]

    init_ax = 0
    axcolor = '#ababab'

    global l
    l = ax.imshow(in_array.take(dims_mean[init_ax], axis=init_ax), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')
    #dot = ax.plot(dims_mean[1], dims_mean[2], 'r+')

    global cursor_on
    global cursor_coord

    cursor_on = True
    cursor_coord = dims_mean[:]

    i_slider_plot = plt.axes([0.25, 0.2, 0.65, 0.03], axisbg='r')
    i_slider = Slider(i_slider_plot, 'i', 0, dims[0] - 1, valinit=dims_mean[0], valfmt='%1i')

    j_slider_plot = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg='g')
    j_slider = Slider(j_slider_plot, 'j', 0, dims[1] - 1, valinit=dims_mean[1], valfmt='%1i')

    k_slider_plot = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='b')
    k_slider = Slider(k_slider_plot, 'k', 0, dims[2] - 1, valinit=dims_mean[2], valfmt='%1i')

    axis_selector_plot = plt.axes([0.02, 0.1, 0.15, 0.13], axisbg=axcolor)
    axis_selector = RadioButtons(axis_selector_plot, ('jk', 'ik', 'ij'), active=0)

    center_image_button_plot = plt.axes([0.02, 0.04, 0.15, 0.04])
    center_image_button = Button(center_image_button_plot, 'Center', color=axcolor, hovercolor='0.975')

    def update_plane(label):

        global l

        if label == 'jk':
            new_i = int(i_slider.val)
            l = ax.imshow(in_array.take(new_i, axis=0), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')
            ax.set_xlim([0, dims[2]])
            ax.set_ylim([0, dims[1]])
            #l.set_array(in_array.take(new_i, axis=0))

        if label == 'ik':
            new_j = int(j_slider.val)
            l = ax.imshow(in_array.take(new_j, axis=1), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')

            ax.set_xlim([0, dims[2]])
            ax.set_ylim([0, dims[0]])
            #l.set_array(in_array.take(new_j, axis=1))

        if label == 'ij':
            new_k = int(k_slider.val)
            l = ax.imshow(in_array.take(new_k, axis=2), aspect='equal', origin='lower', interpolation='nearest', cmap='gray')
            ax.set_xlim([0, dims[1]])
            ax.set_ylim([0, dims[0]])
            #l.set_array(in_array.take(new_k, axis=2))

        fig.canvas.draw()

    def update_slides(val):

        global l
        global cursor_coord

        new_i = int(i_slider.val)
        new_j = int(j_slider.val)
        new_k = int(k_slider.val)

        cursor_coord = [new_i, new_j, new_k]

        if axis_selector.value_selected == 'jk':
            l.set_array(in_array.take(new_i, axis=0))

        if axis_selector.value_selected == 'ik':
            l.set_array(in_array.take(new_j, axis=1))

        if axis_selector.value_selected == 'ij':
            l.set_array(in_array.take(new_k, axis=2))

        fig.canvas.draw_idle()

    def reset_slides(event):
        i_slider.reset()
        j_slider.reset()
        k_slider.reset()

    axis_selector.on_clicked(update_plane)

    i_slider.on_changed(update_slides)
    j_slider.on_changed(update_slides)
    k_slider.on_changed(update_slides)

    center_image_button.on_clicked(reset_slides)



    axis_select_cursor = plt.axes([0.05, 0.4, 0.1, 0.15])
    check = CheckButtons(axis_select_cursor, ('Cursor', ), (cursor_on, ))

    def func(label):
        global cursor_on
        global cursor_coord

        if label == 'Cursor':
            cursor_on = bool((cursor_on + 1) % 2)
            print cursor_on
            print cursor_coord

        plt.draw()
    check.on_clicked(func)


    '''
    def onclick(event):
        ix, iy = event.xdata, event.ydata
        print 'vx = %d, vy = %d' % (ix, iy)

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    '''

    if len(dims) >= 4:

        t_slider_plot = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg='b')
        t_slider = Slider(t_slider_plot, 'k', 0, dims[3], valinit=dims_mean[init_ax], valfmt='%1i')

        def update_t(val):
            new_t = int(t_slider.val)
            l.set_array(in_array.take(new_t, axis=3))
            fig.canvas.draw_idle()

        t_slider.on_changed(update_t)

    plt.show(block=block)

