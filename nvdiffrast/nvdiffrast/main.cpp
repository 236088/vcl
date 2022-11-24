#include "preset.h"
#include <ctime>
struct timespec pre, cur, start;
unsigned int step = 0;
float loss_sum = 0;
bool play = false;

#define PAUSE 1000
#define PRESET_NUM 0

#if PRESET_NUM==0
PresetPrimitives preset;
#elif PRESET_NUM==1
PresetPose preset;
#endif


static void InitFunc()
{
	timespec_get(&start, TIME_UTC);
	srand(start.tv_sec^start.tv_nsec);
	preset.init();
	preset.update(0, 0, step, play);
}

static void DisplayFunc(void)
{
	preset.display();
}

static void IdleFunc(void) 
{	
	if (!play)return;
	step++;
	if (step % PAUSE == 0) {
		play = false;
		return;
	}
	pre = cur;
	timespec_get(&cur, TIME_UTC);
	double dt = double(cur.tv_sec - pre.tv_sec) + double(cur.tv_nsec - pre.tv_nsec) * 1e-9;
	double t = double(cur.tv_sec - start.tv_sec) + double(cur.tv_nsec - start.tv_nsec) * 1e-9;
	preset.update(dt, t, step, play);
	cout 
		<<"step:"<< step
		<<" delta time:" << dt
		<<" fps:" << 1.f/dt
		<<" progress time:"<< t
	<<endl;
	glutPostRedisplay();
}

static void KeyboardFunc(unsigned char key, int x, int y) {
	switch (key)
	{
	case ' ':
		play = !play;
		break;
	case 0x1b:
		exit(0);
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