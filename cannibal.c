#include <stdio.h>
#include <stdlib.h>
#define MAX 1000
typedef struct State
{
 int ml, cl, mr, cr, boat;
 struct State* parent;
} State;
typedef struct Queue
{
 State* data[MAX];
 int front, rear;
} Queue;


int moves[5][2] =
{
 {1, 0}, {2, 0}, {0, 1}, {0, 2}, {1, 1}
};
// Queue operations
void init_queue(Queue* q)
{
 q->front = q->rear = 0;
}
void enqueue(Queue* q, State* s)
{
 q->data[q->rear++] = s;
}
State* dequeue(Queue* q)
{
 return q->data[q->front++];
}
int is_empty(Queue* q)
{
 return q->front == q->rear;
}
int is_valid(State* s)
{
 if (s->ml < 0 || s->cl < 0 || s->mr < 0 || s->cr < 0) return 0;
 if (s->ml > 3 || s->cl > 3 || s->mr > 3 || s->cr > 3) return 0;
 if ((s->ml > 0 && s->ml < s->cl) || (s->mr > 0 && s->mr < s->cr)) return 0;
 return 1;
}
int is_equal(State* a, State* b)
{
 return a->ml == b->ml && a->cl == b->cl &&
 a->mr == b->mr && a->cr == b->cr &&
 a->boat == b->boat;

 }
int is_goal(State* s)
{
 return s->ml == 0 && s->cl == 0 &&
 s->mr == 3 && s->cr == 3 && s->boat == 0;
}
int visited_contains(State* visited[], int count, State* s)
{
 for (int i = 0; i < count; i++) {
 if (is_equal(visited[i], s)) return 1;
 }
 return 0;
}
State* create_state(int ml, int cl, int mr, int cr, int boat, State* parent)
{
 State* s = (State*)malloc(sizeof(State));
 s->ml = ml; s->cl = cl; s->mr = mr; s->cr = cr; s->boat = boat;
 s->parent = parent;
 return s;
}
void print_path(State* s)
{
 if (s == NULL) return;
 print_path(s->parent);
 static int step = 1;
 printf("Step %2d: Left -> M: %d, C: %d | Right -> M: %d, C: %d | Boat on
%s bank\n",
 step++, s->ml, s->cl, s->mr, s->cr, s->boat ? "Left" : "Right");
}
void bfs() {
 Queue q;
 init_queue(&q);
 State* visited[MAX];

 int visited_count = 0;
 State* start = create_state(3, 3, 0, 0, 1, NULL);
 enqueue(&q, start);
 visited[visited_count++] = start;
 while (!is_empty(&q))
 {
 State* current = dequeue(&q);
 if (is_goal(current))
 {
 printf("\n Solution found!\n\n");
 print_path(current);
 return;
 }
 for (int i = 0; i < 5; i++) {
 int m = moves[i][0], c = moves[i][1];
 State* new_state;
 if (current->boat == 1)
 {
 new_state = create_state(current->ml - m, current->cl - c,
 current->mr + m, current->cr + c,
 0, current);
 }
 else
 {
 new_state = create_state(current->ml + m, current->cl + c,
 current->mr - m, current->cr - c,
 1, current);
 }
 if (is_valid(new_state) && !visited_contains(visited, visited_count,
new_state)) {
 enqueue(&q, new_state);
 visited[visited_count++] = new_state;
 } else {

    free(new_state); // Clean up invalid or duplicate states
 }
 }
 }
 printf("No solution found.\n");
}
int main()
{
 bfs();
 return 0;
}


// gcc exp1.c
// .\a.exe