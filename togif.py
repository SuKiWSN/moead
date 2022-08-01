import imageio
import os


def create_gif(image_list, gif_name, duration=0.1):
    '''
    :param image_list: 这个列表用于存放生成动图的图片
    :param gif_name: 字符串，所生成gif文件名，带.gif后缀
    :param duration: 图像间隔时间
    :return:
    '''
    frames = []
    for image_name in image_list:
        frames.append(imageio.imread(image_name))

    imageio.mimsave(gif_name, frames, 'GIF', duration=duration)
    return


def main():

    fl = os.listdir('./gif')
    fl = [int(i.split('.')[0]) for i in fl]
    fl.sort()
    fselect = []
    for f in fl:
        if f <= 20:
            fselect.append(f)
        elif f <= 2000:
            if f % 100 == 0:
                fselect.append(f)
        else:
            if f % 2000 == 0:
                fselect.append(f)
    for i in range(20):
        fselect.append(fl[-1])
    image_list = ['./gif/' + str(i) + '.png' for i in fselect]
    gif_name = 'res.gif'
    duration = 0.1
    create_gif(image_list, gif_name, duration)

if __name__=='__main__':
    main()