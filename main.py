import sys
import math
import heapq # Heap binária mínima
import os
from collections import defaultdict

try:
    import graphviz
except ImportError:
    graphviz = None
    print("Graphviz lib isn't installed. Run 'pip install graphviz' to enable visualization.\n")

# Leitura de grafo DOT simples


def read_dot_file(path): 
  """
  Lê um grafo no formato DOT simples (sem atributos).
  Retorna (is_directed, edges_list, vertices_list)
  """
  with open(path, 'r', encoding='utf-8') as f:
    content = f.readlines()
    
  is_directed = 'digraph' in content[0]
  delimiter = '->' if is_directed else '--'
  content = [line.replace('\n', '').replace(';', '').strip().split(delimiter) for line in content[1:-1]]
  edges = [] 
  for line in content: 
    for i in range(len(line) - 1): 
      line[i+1] = line[i+1].strip().split(' [label=')
      if len(line[i+1]) == 2:
        line[i+1][-1] = line[i+1][-1][:-1]
      else:
        # Se não houver um peso atribuído no arquivo, será 1
        line[i+1].append(1)
      edges.append((line[i].strip(), delimiter, line[i+1][0], line[i+1][1]))
  
  vertices = set() 
  edges_list = [] 
  for u, type, v, w in edges:
    edges_list.append((u, v, int(w)))
    vertices.update([u, v])
    if type == '--' and not is_directed: # não-direcionado: adicionar aresta inversa
      # Importante para execução do Floyd-Warshall
      edges_list.append((v, u, int(w)))
    if (type == '--' and is_directed) or (type == '->' and not is_directed):
      raise AssertionError("Graph type and edges connections in file don't match.") 
    
  return is_directed, edges_list, sorted(vertices)


# Algoritmo de Prim (Árvore Geradora Mínima)

def prim(vertices, edges):
  if not vertices:
    return []

  adj = defaultdict(list)
  for u, v, w in edges:
    adj[u].append((v, w))
    adj[v].append((u, w))

  root = vertices[0]
  min_weight = {v: math.inf for v in vertices}
  parent = {v: None for v in vertices}
  min_weight[root] = 0

  pq = [(0, root)]
  in_tree = set()

  while pq:
    k, u = heapq.heappop(pq)
    if u in in_tree:
      continue
    in_tree.add(u)

    for v, w in adj[u]:
      if v not in in_tree and w < min_weight[v]:
        min_weight[v] = w
        parent[v] = u
        heapq.heappush(pq, (w, v)) 

  mst_edges = [(parent[v], v, min_weight[v]) for v in vertices if parent[v] is not None]
  total_weight = sum(min_weight[v] for v in vertices if parent[v] is not None)
  return mst_edges, total_weight


# Algoritmo de Bellman-Ford

def bellman_ford(vertices, edges, source):
  dist = {v: math.inf for v in vertices}
  parent = {v: None for v in vertices}
  dist[source] = 0

  for _ in range(len(vertices) - 1):
    updated = False
    for u, v, w in edges:
      if dist[u] + w < dist[v]:
        dist[v] = dist[u] + w
        parent[v] = u
        updated = True
    if not updated:
      break # Houve uma iteração sem relaxamento, logo se houverem mais elas também não terão 

  # Verificar ciclos negativos
  for u, v, w in edges:
    if dist[u] + w < dist[v]:
      raise ValueError("Grafo contém ciclo negativo.")

  return dist, parent


# Algoritmo de Floyd-Warshall

def floyd_warshall(vertices, edges):
  n = len(vertices)
  pos = {v: i for i, v in enumerate(vertices)}
  dist = [[math.inf]*n for _ in range(n)]
  next_node = [[None]*n for _ in range(n)]

  for v in vertices:
    dist[pos[v]][pos[v]] = 0
    next_node[pos[v]][pos[v]] = v
  for u, v, w in edges:
    dist[pos[u]][pos[v]] = min(dist[pos[u]][pos[v]], w)
    next_node[pos[u]][pos[v]] = v

  for k in range(n):
    for i in range(n):
      for j in range(n):
        if dist[i][j] > dist[i][k] + dist[k][j]:
          dist[i][j] = dist[i][k] + dist[k][j]
          next_node[i][j] = next_node[i][k]

  return dist, pos, next_node


# Visualização com Graphviz

def generate_graph(is_directed, edges, output_path="graph.png"):
    if graphviz is None:
        print("Graphviz not available. Install with 'pip install graphviz' and verify if Graphviz is in PATH.")
        return

    dot = graphviz.Digraph(comment="Grafo") if is_directed else graphviz.Graph(comment="Grafo")
    for u, v, w in edges:
        dot.edge(u, v, label=str(w))
    dot.render(filename=output_path, format='png', cleanup=True)
    print(f"Grafo gerado: {output_path}.png")


def visualize_mst(mst_edges, name="mst_tree"):
    if graphviz is None:
        print("Graphviz not available. Install with 'pip install graphviz'.")
        return

    dot = graphviz.Graph(comment="Árvore Geradora Mínima")
    for u, v, w in mst_edges:
        dot.edge(u, v, label=str(w))
    dot.render(filename=name, format='png', cleanup=True)
    print(f"Árvore Geradora Mínima gerada: {name}.png")


def visualize_bellman_paths(edges, parent, source, name="bellman_paths"):
  if graphviz is None:
    return
  dot = graphviz.Digraph(comment="Caminhos Bellman-Ford")
  used = set()
  for v, p in parent.items():
    if p is not None:
      used.add((p, v))
  for u, v, w in edges:
    color = "red" if (u, v) in used else "gray"
    penwidth = "2" if (u, v) in used else "1"
    dot.edge(u, v, label=str(w), color=color, penwidth=penwidth)
  dot.render(filename=name, format='png', cleanup=True)
  print(f"Caminhos Bellman-Ford gerados: {name}.png")


def reconstruct_path(u, v, vertices, pos, next_node):
  """Reconstrói o caminho mínimo entre u e v a partir da matriz next_node."""
  if next_node[pos[u]][pos[v]] is None:
    return []
  path = [u]
  while u != v:
    u = next_node[pos[u]][pos[v]]
    path.append(u)
  return path


def visualize_floyd_paths_per_source(vertices, edges, dist, pos, next_node, name_prefix="floyd_paths"):
  """
  Gera uma imagem por vértice de origem, destacando os caminhos mínimos
  a partir desse vértice com base nas matrizes de Floyd-Warshall.
  """
  if graphviz is None:
    print("Graphviz not available. Install with 'pip install graphviz'.")
    return

  for i, source in enumerate(vertices):
    used_edges = set()

    # Reconstroi todos os caminhos que saem do vértice 'source'
    for j, target in enumerate(vertices):
      if source == target or dist[i][j] == math.inf:
        continue
      path = reconstruct_path(source, target, vertices, pos, next_node)
      for k in range(len(path) - 1):
        used_edges.add((path[k], path[k + 1]))

    # Cria um gráfico para essa origem
    dot = graphviz.Digraph(comment=f"Caminhos mínimos a partir de {source}")
    dot.node(source, color="red", style="filled", fillcolor="#ffcccc")
    for u, v, w in edges:
      if (u, v) in used_edges:
        dot.edge(u, v, label=str(w), color="green", penwidth="2")
      else:
        dot.edge(u, v, label=str(w), color="gray", penwidth="1")

    file_name = f"{name_prefix}_{source}"
    dot.render(filename=file_name, format="png", cleanup=True)
    print(f"Caminhos Floyd-Warshall a partir de '{source}' gerados: {file_name}.png")


# main

def main():
  if len(sys.argv) < 2:
    print("Uso: python trabalho2.py <arquivo.dot> [--plot]")
    sys.exit(1)

  path = sys.argv[1]
  plot = len(sys.argv) == 3 and sys.argv[2] == "--plot"

  is_directed, edges, vertices = read_dot_file(path)

  print(f"Grafo {'direcionado' if is_directed else 'não direcionado'}")
  print("Vértices:", vertices)
  print("Arestas:", edges)

  # Prim (apenas grafos não direcionados)
  if not is_directed:
    mst_edges, total_weight = prim(vertices, edges)
    print("\n=== Algoritmo de Prim ===")
    print("Arestas da AGM:", mst_edges)
    print(f"Peso total da árvore: {total_weight}")

  # Bellman-Ford
  source = vertices[0]
  print("\n=== Algoritmo de Bellman-Ford ===")
  try:
    dist, parent = bellman_ford(vertices, edges, source)
    for v in vertices:
      print(f"{source} -> {v}: dist = {dist[v]}, pai = {parent[v]}")
  except ValueError as e:
    print("Erro:", e)

  # Floyd-Warshall
  print("\n=== Algoritmo de Floyd-Warshall ===")
  dist, pos, next_node = floyd_warshall(vertices, edges)
  print("Matriz de distâncias:")
  print("    ", " ".join(vertices))
  for i, u in enumerate(vertices):
    row = " ".join(f"{dist[i][pos[v]] if dist[i][pos[v]] != math.inf else '∞':>5}" for v in vertices)
    print(f"{u:>3} {row}")


  # Visualização
  if plot:
    if not os.path.exists('./output'):
      os.mkdir('output')
    base = os.path.splitext(os.path.basename(path))[0]
    if not os.path.exists(f'./output/{base}'):
      os.mkdir(f'./output/{base}')
    os.chdir(f'./output/{base}')
    generate_graph(is_directed, edges, output_path=base)
    if not is_directed:
      visualize_mst(mst_edges)
    visualize_bellman_paths(edges, parent, source)
    visualize_floyd_paths_per_source(vertices, edges, dist, pos, next_node)

if __name__ == "__main__":
  main()
