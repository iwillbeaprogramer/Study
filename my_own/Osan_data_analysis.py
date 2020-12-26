class Grid:
    def __init__(self, _lon=0,_lat=0, _Population = 0, _YoungPopulation = 0,_PrimaryKey=0,_is_Osan=0):
        self.Dong=''     # 해당 격자가 속해있는 동의 이름
        self.is_Osan=_is_Osan     # Grid_class를 만들때 경계에 있는 격자를 처리하기 위한 변수
        self.PrimaryKey=_PrimaryKey     # Grid 객체마다 고유의 숫자를 부여하여 중복을 피함
        self.Lon=_lon      #Longitude , x , 경도
        self.Lat=_lat      #Latitude , y , 위도
        self.Population = _Population # 거주인구수
        self.YoungPopulation=_YoungPopulation # 어린이
    
    # 프린트함수
    
    # 등위 연산자 정의
    def __eq__(self,G):
        if (self.Lon==G.Lon) and (self.Lat==G.Lat):
            return True
        else:
            return False
    
x_min,y_min = 126995, 37125
x_max,y_max = 127100, 37200
not_Osan=[]     #후에 안양시 경계에 놓여있는 격자를 처리하기 위한 변수
Grid_list=[]     # 안양시 지도를 일정 단위로 나누어 저장한 Grid 들을 담고있는 list
count=0     # PrimaryKey를 부여하기위해 놓은 변수

# 격자를 먼저 만들어놓고 추후에 값들을 넣고 필요없는 격자를 빼는 식으로 한다.
for i in range(y_min,y_max):
    for j in range(x_min,x_max):
        temp=Grid(j/1000,i/1000,_Population=0,_YoungPopulation=0,_PrimaryKey=count)
        Grid_list.append(temp)
        count+=1


# Population 함수
# 해당 위도 경도에 맞는 격자를 찾아서 반환
def Search_index(temp):
    # input = [위도,경도]
    grid=Grid(_floor(temp[0]),_floor(temp[1]))
    try :
        # 위도 경도로 검색한 격자가 있을 경우 해당 인덱스를 반환
        result = Grid_list.index(grid)
        return result
    except :
        # 위도 경도로 검색한 격자가 없을 경우 -1을 리턴
        return -1

# grid.Population 의 변수에 인구수를 더해줌
def P_Grid(temp):
    index = Search_index(temp)
    Grid_list[index].Population+=temp[2]
    Grid_list[index].is_Anyang+=1
    return

def older_Grid(temp):
    index = Search_index(temp)
    Grid_list[index].old_people+=temp[2]
    Grid_list[index].is_Anyang+=1
    return

def youth_Grid(temp):
    index = Search_index(temp)
    Grid_list[index].youth+=temp[2]
    Grid_list[index].is_Anyang+=1
    return

# Grid_list에 있는 모든 grid에 인구수를 전부 더해줌
def input_Population(temp):
    for i in range(len(temp)):
        P_Grid(temp[i])
        
def input_older(temp):
    for i in range(len(temp)):
        older_Grid(temp[i])
        
def input_youth(temp):
    for i in range(len(temp)):
        youth_Grid(temp[i])
