#!/usr/bin/env python
# -*- coding: UTF-8 -*-

"""
-------------------------------------------------------------------------
This file is part of the MindStudio project.
Copyright (c) 2026 Huawei Technologies Co.,Ltd.

MindStudio is licensed under Mulan PSL v2.
You can use this software according to the terms and conditions of the Mulan PSL v2.
You may obtain a copy of Mulan PSL v2 at:

         http://license.coscl.org.cn/MulanPSL2

THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
See the Mulan PSL v2 for more details.
-------------------------------------------------------------------------

msmodelslim.utils.security.request 模块的单元测试
"""

from unittest.mock import MagicMock, patch

import pytest

from msmodelslim.utils.exception import SecurityError
from msmodelslim.utils.security import request as req_module


class TestIsAllowedIp:
    """测试 _is_allowed_ipv4 / _is_allowed_ipv6 内网与回环地址判定"""

    def test_is_allowed_ipv4_return_true_when_loopback(self):
        """IPv4 回环地址应判定为允许"""
        assert req_module._is_allowed_ipv4("127.0.0.1") is True

    def test_is_allowed_ipv4_return_true_when_private(self):
        """RFC1918 私网 IPv4 应判定为允许"""
        assert req_module._is_allowed_ipv4("10.0.0.1") is True
        assert req_module._is_allowed_ipv4("192.168.1.1") is True

    def test_is_allowed_ipv4_return_false_when_public(self):
        """公网 IPv4 应判定为不允许"""
        assert req_module._is_allowed_ipv4("8.8.8.8") is False

    def test_is_allowed_ipv4_return_false_when_invalid(self):
        """非法 IPv4 字符串应判定为不允许"""
        assert req_module._is_allowed_ipv4("not-an-ip") is False

    def test_is_allowed_ipv6_return_true_when_loopback(self):
        """IPv6 回环地址应判定为允许"""
        assert req_module._is_allowed_ipv6("::1") is True

    def test_is_allowed_ipv6_return_true_when_link_local(self):
        """IPv6 链路本地地址应判定为允许"""
        assert req_module._is_allowed_ipv6("fe80::1") is True

    def test_is_allowed_ipv6_return_false_when_public(self):
        """公网 IPv6 应判定为不允许"""
        assert req_module._is_allowed_ipv6("2001:4860:4860::8888") is False

    def test_is_allowed_ipv6_return_false_when_invalid(self):
        """非法 IPv6 字符串应判定为不允许"""
        assert req_module._is_allowed_ipv6("bad-ipv6") is False


class TestNormalizeHost:
    """测试 _normalize_host 主机名标准化"""

    def test_normalize_host_return_compressed_when_ipv6(self):
        """IPv6 应压缩为规范短格式"""
        assert req_module._normalize_host("2001:0db8:0000:0000:0000:ff00:0042:8329") == "2001:db8::ff00:42:8329"

    def test_normalize_host_return_unchanged_when_hostname(self):
        """普通主机名应原样返回"""
        assert req_module._normalize_host("localhost") == "localhost"


class TestValidateSafeHost:
    """测试 validate_safe_host SSRF 主机校验"""

    def test_validate_safe_host_raises_when_empty(self):
        """空 host 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="cannot be empty"):
            req_module.validate_safe_host("")

    def test_validate_safe_host_return_localhost_when_localhost(self):
        """localhost 应通过校验"""
        assert req_module.validate_safe_host("localhost") == "localhost"

    def test_validate_safe_host_return_loopback_when_127(self):
        """127.0.0.1 应通过校验"""
        assert req_module.validate_safe_host("127.0.0.1") == "127.0.0.1"

    def test_validate_safe_host_return_ipv6_loopback_when_bracketed(self):
        """带方括号的 [::1] 应规范为 ::1"""
        assert req_module.validate_safe_host("[::1]") == "::1"

    def test_validate_safe_host_return_private_when_rfc1918(self):
        """内网 IP 应通过校验"""
        assert req_module.validate_safe_host("10.1.2.3") == "10.1.2.3"

    def test_validate_safe_host_raises_when_public_ip(self):
        """公网 IP 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="not allowed"):
            req_module.validate_safe_host("8.8.8.8")

    def test_validate_safe_host_return_host_when_dns_resolves_private(self):
        """DNS 解析到内网地址的主机名应通过校验"""
        with patch("socket.getaddrinfo") as mock_gai:
            mock_gai.return_value = [
                (2, 1, 6, "", ("192.168.0.10", 0)),
            ]
            assert req_module.validate_safe_host("internal.host") == "internal.host"

    def test_validate_safe_host_raises_when_dns_unresolvable(self):
        """无法解析或解析到非公网允许地址时应抛出 SecurityError"""
        with patch("socket.getaddrinfo", side_effect=OSError("fail")):
            with pytest.raises(SecurityError, match="not allowed"):
                req_module.validate_safe_host("evil.example.com")


class TestValidateSafeEndpoint:
    """测试 validate_safe_endpoint 路径遍历防护"""

    def test_validate_safe_endpoint_raises_when_empty(self):
        """空 endpoint 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="cannot be empty"):
            req_module.validate_safe_endpoint("")

    def test_validate_safe_endpoint_raises_when_no_leading_slash(self):
        """非绝对路径应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="must start with"):
            req_module.validate_safe_endpoint("api/v1")

    def test_validate_safe_endpoint_raises_when_path_traversal(self):
        """含 .. 的路径应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="invalid path"):
            req_module.validate_safe_endpoint("/api/../secret")

    def test_validate_safe_endpoint_raises_when_double_slash(self):
        """以 // 开头的路径应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="invalid path"):
            req_module.validate_safe_endpoint("//api")

    def test_validate_safe_endpoint_raises_when_invalid_chars(self):
        """含非法字符的路径应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="invalid characters"):
            req_module.validate_safe_endpoint("/api?x=1")

    def test_validate_safe_endpoint_return_path_when_valid(self):
        """合法绝对路径应原样返回"""
        assert req_module.validate_safe_endpoint("/v1/models") == "/v1/models"


class TestBuildSafeUrl:
    """测试 build_safe_url 安全 URL 拼接"""

    def test_build_safe_url_return_http_url_when_localhost(self):
        """回环地址应拼出正确的 http URL"""
        url = req_module.build_safe_url("127.0.0.1", 8080, "/health")
        assert url == "http://127.0.0.1:8080/health"

    def test_build_safe_url_wrap_ipv6_when_ipv6_host(self):
        """IPv6 host 在 URL 中应使用方括号包裹"""
        url = req_module.build_safe_url("::1", 8000, "/ping")
        assert url == "http://[::1]:8000/ping"

    def test_build_safe_url_raises_when_invalid_scheme(self):
        """非 http/https scheme 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="Invalid URL scheme"):
            req_module.build_safe_url("127.0.0.1", 80, "/x", scheme="ftp")


class TestSafeGet:
    """测试 safe_get 安全 GET 请求"""

    def test_safe_get_return_response_when_valid_url(self):
        """合法内网 URL 应成功发起 GET"""
        mock_resp = MagicMock()
        with patch("requests.get", return_value=mock_resp) as mock_get:
            resp = req_module.safe_get("http://127.0.0.1:8080/health")
        assert resp is mock_resp
        mock_get.assert_called_once()

    def test_safe_get_raises_when_invalid_scheme(self):
        """非法 scheme 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="Invalid URL scheme"):
            req_module.safe_get("file:///etc/passwd")

    def test_safe_get_raises_when_unsafe_hostname(self):
        """不安全 hostname 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="not safe"):
            req_module.safe_get("http://8.8.8.8/health")

    def test_safe_get_wrap_error_when_validate_host_fails(self):
        """validate_safe_host 失败时应包装为 SecurityError 再抛出"""
        with patch.object(req_module, "validate_safe_host", side_effect=SecurityError("bad host")):
            with pytest.raises(SecurityError, match="not safe"):
                req_module.safe_get("http://127.0.0.1/x")


class TestSafePost:
    """测试 safe_post 安全 POST 请求"""

    def test_safe_post_return_response_when_valid_url(self):
        """合法内网 URL 应成功发起 POST"""
        mock_resp = MagicMock()
        with patch("requests.post", return_value=mock_resp) as mock_post:
            resp = req_module.safe_post("http://127.0.0.1:8080/submit", json={"k": "v"})
        assert resp is mock_resp
        mock_post.assert_called_once()

    def test_safe_post_raises_when_invalid_scheme(self):
        """非法 scheme 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="Invalid URL scheme"):
            req_module.safe_post("gopher://127.0.0.1/x")

    def test_safe_post_raises_when_unsafe_hostname(self):
        """不安全 hostname 应抛出 SecurityError"""
        with pytest.raises(SecurityError, match="not safe"):
            req_module.safe_post("http://1.2.3.4/x")
