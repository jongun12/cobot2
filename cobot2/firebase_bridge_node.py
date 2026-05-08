import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32

# Firebase Admin SDK 임포트
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)

class FirebaseBridgeNode(Node):
    def __init__(self):
        super().__init__('firebase_bridge_node')
        
        # 1. Firebase 초기화
        # 주의: 다운로드 받은 JSON 키 파일의 실제 절대 경로로 변경하세요.
        key_path = f'{PACKAGE_PATH}/serviceAccountKey.json'
        key_path = '/home/kim/cobot_ws/src/cobot2/resource/serviceAccountKey.json'
        
        try:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.get_logger().info('Firebase 초기화 성공!')
        except Exception as e:
            self.get_logger().error(f'Firebase 초기화 실패: {e}')
            return

        # 2. ROS 2 Subscriber 생성
        # 'robot_status'라는 토픽에서 String 메시지를 받습니다.
        self.subscription = self.create_subscription(
            Int32,
            'trash_count',
            self.status_callback,
            10
        )
        self.get_logger().info('브릿지 노드가 실행되었습니다. 데이터를 대기 중입니다...')

    def status_callback(self, msg):
        can, plastic, paper = 0, 0, 0
        if msg.data == 0:
            plastic = 1
        elif msg.data == 1:
            plastic = 1
        elif msg.data == 2:
            plastic = 1
        elif msg.data == 3:
            plastic = 1
        elif msg.data == 4:
            can = 1
        elif msg.data == 5:
            paper = 1
        
        self.get_logger().info(f'ROS 2에서 데이터 수신: "{msg.data}"')
        
        # 3. Firebase Firestore에 데이터 쓰기
        try:
            # 'trash_limit' 컬렉션 안의 'first_task' 문서에 데이터를 업데이트합니다.
            doc_ref = self.db.collection('trash_limit').document('TvQe4stbmyYjewO8tKck')
            doc_ref.set({
                'can': firestore.Increment(can),
                'plastic': firestore.Increment(plastic),
                'paper': firestore.Increment(paper)
            }, merge=True) # merge=True를 사용해 기존 데이터를 덮어쓰지 않고 업데이트
            
            self.get_logger().info('Firebase에 데이터 전송 완료')
        except Exception as e:
            self.get_logger().error(f'Firebase 전송 오류: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = FirebaseBridgeNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('사용자에 의해 노드가 종료되었습니다.')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()