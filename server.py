from flask import Flask, render_template, Response
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import io

app = Flask(__name__)

# list_file 내용 가져오기
with open('list_file.txt', 'r') as f:
    list_file = open('list_file.txt', 'r').read().split('\n')

for i in range(len(list_file) - 1):
    list_file[i] = float(list_file[i])
list_file.pop()  # 공백 제거
avg_concentration = float(list_file.pop())
avg_concentration = round(avg_concentration, 4)
concentrate_list = list_file

# index_file 내용 가져오기
with open('index_file.txt', 'r') as f:
    index_file = open('index_file.txt', 'r').read().split('\n')

for i in range(len(index_file) - 1):
    index_file[i] = int(index_file[i])
index_file.pop()
index_list = index_file


def get_emoji():
    if avg_concentration > 85:
        emoji = '😊'
    elif avg_concentration > 60:
        emoji = '😐'
    else:
        emoji = '🙄'
    return emoji


# 집중도 그래프 생성
def concentrate_plot(con_list):
    fig = Figure(figsize=(12, 5))
    start = int(index_list[0]) - 1
    end = index_list[1]
    axis = fig.add_subplot(1, 1, 1)
    # 집중도
    xs = range(len(con_list))
    ys = con_list

    axis.plot(xs, ys, color='#534847', linewidth=2)
    axis.set_xlabel("sec")
    axis.set_ylabel("Concentration")
    axis.axvspan(start, end, facecolor='#EE7785', alpha=0.2)

    return fig


@app.route('/')
def show_main(num=None):
    #TODO 집중도 가져오기
    emoji = get_emoji()
    return render_template('main.html', num=num, emoji=emoji, start=(int(index_list[0]) - 1), end=index_list[1],
                           avg_concentration=avg_concentration)


@app.route('/concentrate.png')
def show_concentrate_plot():
    # TODO 집중도 받아오기
    fig = concentrate_plot(concentrate_list)
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return Response(output.getvalue(), mimetype='image/png')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=7000)
