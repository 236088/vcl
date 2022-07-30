#include "preset.h"
#include <ctime>
struct timespec pre, cur, start;
float loss_sum = 0;
bool play = false;

#define PRESET 0

#if PRESET==1
PresetEarth preset;
#elif PRESET==2
PresetCube preset;
#elif PRESET==3
PresetPhong preset;
#elif PRESET==4
PresetFilter preset;
#else
PresetPrimitives preset;
#endif


static void InitFunc()
{
	timespec_get(&start, TIME_UTC);
	srand(0);
	preset.init();
}

static void DisplayFunc(void)
{
	preset.display();
}

static void IdleFunc(void) 
{	
	if (!play)return;
	pre = cur;
	timespec_get(&cur, TIME_UTC);
	double dt = double(cur.tv_sec - pre.tv_sec) + double(cur.tv_nsec - pre.tv_nsec) * 1e-9;
	double t = double(cur.tv_sec - start.tv_sec) + double(cur.tv_nsec - start.tv_nsec) * 1e-9;
	preset.update(dt,t, play);

	glutPostRedisplay();
}

//スペースキーでスタート/ポーズ
static void KeyboardFunc(unsigned char key, int x, int y) {
	switch (key)
	{
	case ' ':
		play = !play;
		break;
	default:
		break;
	}
}

int main(int argc, char* argv[])
{
	glutInit(&argc, argv);
	glutInitWindowSize(preset.windowWidth, preset.windowHeight);
	glutInitDisplayMode(GLUT_RGBA);
	glutCreateWindow(argv[0]);
	glewInit();
	InitFunc();
	glutDisplayFunc(DisplayFunc);
	glutIdleFunc(IdleFunc);
	glutKeyboardFunc(KeyboardFunc);
	glutMainLoop();
}