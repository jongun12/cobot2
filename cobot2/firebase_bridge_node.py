import os

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32
from std_srvs.srv import Trigger
from dsr_msgs2.srv import MoveStop

# Firebase Admin SDK 임포트
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
from ament_index_python.packages import get_package_share_directory

PACKAGE_NAME = "cobot2"
PACKAGE_PATH = get_package_share_directory(PACKAGE_NAME)
DEFAULT_KEY_PATH = os.path.join(PACKAGE_PATH, "resource", "serviceAccountKey.json")
ROBOT_ID = "dsr01"

class FirebaseBridgeNode(Node):
    def __init__(self):
        super().__init__('firebase_bridge_node')
        
        # 1. Firebase 초기화
        key_path = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY", DEFAULT_KEY_PATH)
        if not os.path.exists(key_path):
            self.get_logger().error(
                f'Firebase 서비스 계정 키 파일을 찾을 수 없습니다: {key_path}'
            )
            self.get_logger().error(
                'resource/serviceAccountKey.json에 두거나 '
                'FIREBASE_SERVICE_ACCOUNT_KEY 환경변수로 경로를 지정하세요.'
            )
            return
        
        try:
            cred = credentials.Certificate(key_path)
            firebase_admin.initialize_app(cred)
            self.db = firestore.client()
            self.get_logger().info(f'Firebase 초기화 성공: {key_path}')
        except Exception as e:
            self.get_logger().error(f'Firebase 초기화 실패: {e}')
            return

        self.trash_doc_ref = self.db.collection('trash_limit').document('TvQe4stbmyYjewO8tKck')
        self.start_condition_ref = self.db.collection('start').document('condition')
        self.cached_flag = 0
        self.last_start_condition = None
        self.last_emergency_stop = None
        self.publisher = self.create_publisher(Int32, 'start_condition', 10)
        self.emergency_stop_publisher = self.create_publisher(
            Int32,
            'emergency_stop',
            10,
        )
        self.move_stop_client = self.create_client(
            MoveStop,
            f'/{ROBOT_ID}/motion/move_stop',
        )
        self.flag_watch = self.trash_doc_ref.on_snapshot(self.flag_snapshot_callback)
        self.start_watch = self.start_condition_ref.on_snapshot(
            self.start_condition_snapshot_callback
        )

        # 2. ROS 2 Subscriber 생성
        # 'robot_status'라는 토픽에서 String 메시지를 받습니다.
        self.subscription = self.create_subscription(
            Int32,
            'trash_count',
            self.status_callback,
            10
        )
        self.task_complete_subscription = self.create_subscription(
            Int32,
            'task_complete',
            self.task_complete_callback,
            10
        )
        self.get_logger().info('브릿지 노드가 실행되었습니다. 데이터를 대기 중입니다...')

        self.get_flag_service = self.create_service(
            Trigger,
            'is_trash_full',
            self.handle_get_flag_service
        )

    def start_condition_snapshot_callback(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if not doc.exists:
                self.get_logger().warn('start/condition 문서가 존재하지 않습니다.')
                continue

            data = doc.to_dict() or {}
            start_condition = int(data.get('condition', 0))
            emergency_stop = int(data.get('emergency_stop', 0))
            if self.last_start_condition != start_condition:
                self.last_start_condition = start_condition
                msg = Int32()
                msg.data = start_condition
                self.publisher.publish(msg)
                self.get_logger().info(f'start_condition 값 발행: {msg.data}')

            if self.last_emergency_stop != emergency_stop:
                self.last_emergency_stop = emergency_stop
                msg = Int32()
                msg.data = emergency_stop
                self.emergency_stop_publisher.publish(msg)
                self.get_logger().info(f'emergency_stop 값 발행: {msg.data}')
                if emergency_stop == 1:
                    self.request_move_stop()

    def request_move_stop(self):
        if not self.move_stop_client.service_is_ready():
            self.move_stop_client.wait_for_service(timeout_sec=0.2)
        if not self.move_stop_client.service_is_ready():
            self.get_logger().error('/dsr01/motion/move_stop 서비스를 사용할 수 없습니다.')
            return

        request = MoveStop.Request()
        request.stop_mode = 0
        self.move_stop_client.call_async(request)
        self.get_logger().warn('/dsr01/motion/move_stop 서비스를 호출했습니다.')

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
            self.trash_doc_ref.set({
                'can': firestore.Increment(can),
                'plastic': firestore.Increment(plastic),
                'paper': firestore.Increment(paper)
            }, merge=True) # merge=True를 사용해 기존 데이터를 덮어쓰지 않고 업데이트
            
            self.get_logger().info('Firebase에 데이터 전송 완료')
        except Exception as e:
            self.get_logger().error(f'Firebase 전송 오류: {e}')

    def task_complete_callback(self, msg):
        if msg.data != 1:
            return

        try:
            self.start_condition_ref.set({'condition': 0}, merge=True)
            self.get_logger().info('작업 완료: start/condition.condition 값을 0으로 변경했습니다.')
        except Exception as e:
            self.get_logger().error(f'start_condition 초기화 오류: {e}')

    def flag_snapshot_callback(self, doc_snapshot, changes, read_time):
        for doc in doc_snapshot:
            if not doc.exists:
                self.get_logger().warn('Firebase flag 문서가 존재하지 않습니다.')
                continue

            data = doc.to_dict()
            self.cached_flag = int(data.get('flag', 0))
            self.get_logger().info(f'Firebase flag 캐시 업데이트: {self.cached_flag}')

    def handle_get_flag_service(self, request, response):
        response.success = True
        response.message = str(self.cached_flag)
        self.get_logger().info(f'캐시된 flag 값 반환: {self.cached_flag}')
        return response


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
