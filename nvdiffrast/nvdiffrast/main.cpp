#include "preset.h"
#include <ctime>
#define CONSOLE_INTERVAL 100
#define PAUSE_COUNT 3000
#define EXIT_COUNT 3000
struct timespec pre, cur;
double t;
float loss_sum = 0;
int count = 0;
bool play = false;

PresetPrimitives preset;
//PresetCube preset;
//PresetEarth preset;
//PresetFilter preset;
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
	loss_sum += preset.getLoss();
	if ((++count) % CONSOLE_INTERVAL == 0) {
		printf(" count: %d, loss: %f\n", count, loss_sum / CONSOLE_INTERVAL);
		loss_sum = 0;
	}
	if (count % PAUSE_COUNT == 0)play = false;
	if (count % EXIT_COUNT == 0)exit(0);
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

	printf("\r%3.3f ms:%3.3f fps", dt * 1e3, 1./ dt);
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