from flask import Flask, render_template, Response
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

import io

app = Flask(__name__)


# 집중도 그래프 생성
def concentrate_plot(c_list):
    fig = Figure(figsize=(12, 5))

    axis = fig.add_subplot(1, 1, 1)

    # 집중도
    xs = range(len(c_list))
    ys = c_list
    axis.plot(xs, ys, color='#534847', linewidth=2)

    range_list = low_section(c_list, 30)

    for start, end in range_list:
        axis.axvspan(start, end, facecolor='#EE7785', alpha=0.2)

    return fig


# 집중도 그래프 & 상세 수치 그래프
def detail_plot(c_list, n_list):
    fig = Figure(figsize=(12, 5))

    axis = fig.add_subplot(1, 1, 1)

    color = ['#67D5B5', '#C89EC4', '#84B1ED']

    # 인식 수치
    for i in range(3):
        xs = range(len(n_list[i]))
        ys = n_list[i]
        axis.plot(xs, ys, color=color[i], linestyle=':')

    # 집중도
    xs = range(len(c_list))
    ys = c_list
    axis.plot(xs, ys, color='#534847', linewidth=2)

    range_list = low_section(c_list, 30)

    for start, end in range_list:
        axis.axvspan(start, end, facecolor='#EE7785', alpha=0.2)

    return fig


# 집중도 낮은 구간 리스트에 저장해서 리턴
def low_section(c_list, low):
    range_list = []
    is_started = False

    start = 0
    end = 0
    i = 0

    # 낮은 구간 찾아서 리스트에 저장
    for l in c_list:
        if l < low:
            if not is_started:
                start = i
                is_started = True
            else:
                end = i
        else:
            if end > 0:
                range_list.append((start, end))
                start = 0
                end = 0
                is_started = False

        i += 1

    # 마지막 구간 저장 못했을 경우
    if (start > 0) and is_started:
        range_list.append((start, end))

    return range_list


# 임시 (dummy 집중도 리스트 생성)
def cal_concentrate():
    concentrate_list = []
    with open('list_file.txt', 'r') as f:
        concentrate_list = open('list_file.txt', 'r').read().split('\n')

    for i in range(len(concentrate_list) - 1):
        concentrate_list[i] = float(concentrate_list[i])
    concentrate_list.pop()
    print(concentrate_list)

    # for i in range(30):
    #     concentrate_list.append(np.random.randint(30, 100))
    # for i in range(30):
    #     concentrate_list.append(np.random.randint(0, 30))
    # for i in range(20):
    #     concentrate_list.append(np.random.randint(30, 100))
    # for i in range(20):
    #     concentrate_list.append(np.random.randint(0, 30))

    return concentrate_list


# def create_detail_plot():


@app.route('/')
def show_main(num=None):
    return render_template('main.html', num=num)


@app.route('/concentrate.png')
def show_concentrate_plot():
    # TODO 집중도 받아오기
    concentrate_list = cal_concentrate()

    fig = concentrate_plot(concentrate_list)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


@app.route('/detail.png')
def show_detail_plot():
    # TODO 상세 수치 받아오기
    n_list1 = [np.random.randint(0, 50) for x in range(100)]
    n_list2 = [np.random.randint(0, 50) for x in range(100)]
    n_list3 = [np.random.randint(0, 50) for x in range(100)]

    num_list = [n_list1, n_list2, n_list3]

    # TODO 집중도 받아오기
    concentrate_list = cal_concentrate()

    fig = detail_plot(concentrate_list, num_list)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run()
