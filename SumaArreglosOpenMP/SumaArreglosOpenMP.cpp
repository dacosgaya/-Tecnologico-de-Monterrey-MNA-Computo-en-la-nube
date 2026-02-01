#include <iostream>
#include <cstdlib>
#include <ctime>
#include <omp.h>

#define chunk   100
#define mostrar 20

using std::cout;
using std::cin;
using std::endl;

void imprimeArreglo(int* arreglo, int cuantos)
{
    for (int i = 0; i < cuantos; i++)
        cout << arreglo[i] << " ";
    cout << endl;
}

// Generador pseudoaleatorio simple (LCG) portable para MSVC
static inline unsigned int lcg_next(unsigned int& state)
{
    // Valores clásicos (Numerical Recipes)
    state = 1664525u * state + 1013904223u;
    return state;
}

int main()
{
    cout << "============================================================\n";
    cout << "TAREA 1 - Suma paralela de dos arreglos con OpenMP\n";
    cout << "============================================================\n\n";

    // Pedir tamaño
    int N = 0;
    cout << "[1/7] Ingresa el tamano del arreglo (ej. 1000): ";
    cin >> N;

    if (N <= 0) {
        cout << "ERROR: N debe ser > 0\n";
        return 1;
    }

	// Configuración
    cout << "\n[2/7] Configuracion:\n";
    cout << "    N      = " << N << "\n";
    cout << "    chunk  = " << chunk << " (schedule static por bloques)\n";
    cout << "    mostrar= " << mostrar << " (elementos a imprimir)\n\n";

    // Reservar memoria
    cout << "[3/7] Reservando memoria para a[], b[], c[]...\n";
    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];
    cout << "    OK: memoria reservada.\n\n";

    // Llenado aleatorio
    cout << "[4/7] Llenando a[] y b[] con valores aleatorios...\n";
    srand((unsigned)time(nullptr));
    for (int i = 0; i < N; i++) {
        a[i] = rand() % 100;
        b[i] = rand() % 100;
        c[i] = 0;
    }
    cout << "    OK: arreglos llenados.\n\n";

    // [5/7] Info OpenMP
    cout << "[5/7] Informacion del entorno OpenMP:\n";
    cout << "    Procesadores/cores visibles (omp_get_num_procs): " << omp_get_num_procs() << "\n";
    cout << "    Hilos maximos disponibles (omp_get_max_threads): " << omp_get_max_threads() << "\n";
    cout << "    Nota: OpenMP elegira automaticamente cuantos hilos usar.\n\n";

    cout << "[6/7] Ejecutando suma paralela: c[i] = a[i] + b[i]\n";
    cout << "    Directiva: #pragma omp parallel for schedule(static, " << chunk << ")\n\n";

    // Cuántos hilos se crean realmente, dentro de una región parallel
    int threads_creados = 0;
#pragma omp parallel
    {
#pragma omp single
        threads_creados = omp_get_num_threads();
    }

    int* cuenta_por_hilo = new int[threads_creados];
    int* min_i = new int[threads_creados];
    int* max_i = new int[threads_creados];

    int* sample_idx = new int[threads_creados];
    int* sample_a = new int[threads_creados];
    int* sample_b = new int[threads_creados];
    int* sample_c = new int[threads_creados];
    bool* tiene_muestra = new bool[threads_creados];

    for (int t = 0; t < threads_creados; t++) {
        cuenta_por_hilo[t] = 0;
        min_i[t] = N;
        max_i[t] = -1;
        sample_idx[t] = -1;
        sample_a[t] = sample_b[t] = sample_c[t] = 0;
        tiene_muestra[t] = false;
    }

    // Suma paralela
#pragma omp parallel for schedule(static, chunk) shared(a,b,c,cuenta_por_hilo,min_i,max_i,sample_idx,sample_a,sample_b,sample_c,tiene_muestra,N)
    for (int i = 0; i < N; i++)
    {
        int tid = omp_get_thread_num();

        c[i] = a[i] + b[i];

        cuenta_por_hilo[tid]++;
        if (i < min_i[tid]) min_i[tid] = i;
        if (i > max_i[tid]) max_i[tid] = i;

        // Guardar una muestra aleatoria por hilo (dentro del bloque asignado)
        if (!tiene_muestra[tid]) {

            int start = (i / chunk) * chunk;
            int end = start + chunk - 1;
            if (end >= N) end = N - 1;

            // Seed por hilo + por bloque
            unsigned int state =
                (unsigned int)time(nullptr)
                ^ (unsigned int)(tid * 2654435761u)
                ^ (unsigned int)(start * 101u);

            int span = (end - start + 1);
            int offset = (int)(lcg_next(state) % (unsigned int)span);
            int idx = start + offset;

            sample_idx[tid] = idx;
            tiene_muestra[tid] = true;

        }
    }

    for (int t = 0; t < threads_creados; t++)
    {
        if (cuenta_por_hilo[t] <= 0) continue;

        int idx = sample_idx[t];
        if (idx >= 0 && idx < N)
        {
            sample_a[t] = a[idx];
            sample_b[t] = b[idx];
            sample_c[t] = c[idx];
        }
    }

    cout << ">> OpenMP CREO " << threads_creados << " hilos en esta region.\n";

    int hilos_con_trabajo = 0;
    for (int t = 0; t < threads_creados; t++)
        if (cuenta_por_hilo[t] > 0) hilos_con_trabajo++;

    cout << ">> Hilos que PROCESARON al menos 1 elemento: " << hilos_con_trabajo << "\n";
    cout << "   (Si N es pequeno o chunk es grande, no todos los hilos reciben trabajo.)\n\n";

    // Muestra por hilo
    cout << "[7/7] Muestra del trabajo por hilo:\n\n";

    for (int t = 0; t < threads_creados; t++)
    {
        if (cuenta_por_hilo[t] <= 0) continue;

        cout << "Hilo " << t << " -> elementos: " << cuenta_por_hilo[t]
            << ", rango: [" << min_i[t] << ", " << max_i[t] << "]\n";

        int idx = sample_idx[t];
        cout << "    Ejemplo: c[" << idx << "] = a[" << idx << "] + b[" << idx << "] = "
            << sample_a[t] << " + " << sample_b[t] << " = " << sample_c[t] << "\n\n";
    }

    // Comprobación
    int m = (N < mostrar ? N : mostrar);
    cout << "Comprobacion (primeros " << m << "):\n";
    cout << "a: "; imprimeArreglo(a, m);
    cout << "b: "; imprimeArreglo(b, m);
    cout << "c = (a+b): "; imprimeArreglo(c, m);
    cout << endl;

    // Validación
    cout << "Validando resultado...\n";
    bool ok = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) { ok = false; break; }
    }
    cout << "Validacion: " << (ok ? "OK (c = a+b)" : "ERROR") << "\n\n";

    delete[] a; delete[] b; delete[] c;
    delete[] cuenta_por_hilo; delete[] min_i; delete[] max_i;
    delete[] sample_idx; delete[] sample_a; delete[] sample_b; delete[] sample_c;
    delete[] tiene_muestra;

    cout << "Fin del programa. Presiona Enter para salir...\n";
    cin.ignore();
    cin.get();
    return 0;
}
