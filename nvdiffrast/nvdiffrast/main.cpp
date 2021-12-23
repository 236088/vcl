#include "preset.h"
#include <ctime>
struct timespec pre, cur;
double t;
float loss_sum = 0;
bool play = false;

//PresetPrimitives preset;
//PresetCube preset;
//PresetEarth preset;
PresetFilter preset;
//PresetPhong preset;


static void InitFunc()
{
	timespec_get(&cur, TIME_UTC);
	srand(cur.tv_nsec);
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
	long diff = cur.tv_nsec - pre.tv_nsec;
	if (diff < 0)diff = 1e9 + cur.tv_nsec - pre.tv_nsec;
	double dt = (double)diff * 1e-9;
	t += dt;
	preset.update(dt);

	glutPostRedisplay();
}

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